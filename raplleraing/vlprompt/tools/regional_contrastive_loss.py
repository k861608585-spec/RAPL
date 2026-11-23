"""Stable NR_RAPL regional positive/negative contrastive objectives."""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from RAPL.vlprompt.tools.topk_reduce import topk_logsumexp, topk_logsumexp_shared


class RegionalContrastiveLoss:
    """Compute local regional losses while preserving RAPL's Top-K semantics."""

    def __init__(self, topk: List[int], negative_denom_weight: float = 1.0) -> None:
        self.topk = topk
        self.negative_denom_weight = negative_denom_weight

    def _log_mass(self, sim_pos: Tensor, sim_neg: Tensor, logit_scale: Tensor) -> Dict[str, Tensor]:
        """Return stable log masses for positive classes and shared negatives.

        ``sim_pos`` is ``[B, J, L, C]`` and ``sim_neg`` is ``[B, L, Q]``.
        ``topk_logsumexp`` expects RAPL-native ``[B, L, C, J]``.
        """
        if sim_pos.ndim != 4 or sim_neg.ndim != 3:
            raise ValueError("Expected sim_pos [B,J,L,C] and sim_neg [B,L,Q]")
        if sim_pos.shape[0] == 0 or sim_pos.shape[2] == 0:
            raise ValueError("Regional losses require a non-empty batch and at least one local token")
        if sim_neg.shape[-1] == 0:
            raise ValueError("Regional losses require at least one enabled negative prompt")
        # Float32 logsumexp prevents fp16 overflow/underflow while preserving
        # gradients to the underlying fp16 prompt and projection parameters.
        scale = logit_scale.float()
        positive_mass = topk_logsumexp((sim_pos.float() * scale).permute(0, 2, 3, 1), self.topk)
        negative_mass = topk_logsumexp_shared(sim_neg.float() * scale, self.topk, n_prompts=sim_pos.shape[1])
        return {
            "positive_per_class": positive_mass,  # [B, J, C]
            "negative_per_prompt": negative_mass,  # [B, J, Q]
            "positive_all": torch.logsumexp(positive_mass, dim=-1),
            "negative_all": torch.logsumexp(negative_mass, dim=-1),
        }

    def positive_loss(self, sim_pos: Tensor, sim_neg: Tensor, targets: Tensor, logit_scale: Tensor) -> Dict[str, Tensor]:
        mass = self._log_mass(sim_pos, sim_neg, logit_scale)
        batch, prompts, _ = mass["positive_per_class"].shape
        target_mass = mass["positive_per_class"].gather(
            2, targets.view(batch, 1, 1).expand(-1, prompts, 1)
        ).squeeze(-1)
        if self.negative_denom_weight > 0:
            weighted_negative = mass["negative_all"] + torch.tensor(
                self.negative_denom_weight, device=sim_pos.device, dtype=sim_pos.dtype
            ).log()
            denominator = torch.logaddexp(mass["positive_all"], weighted_negative)
        else:
            denominator = mass["positive_all"]
        loss = -(target_mass - denominator).mean()
        predictions = mass["positive_per_class"].mean(dim=1).argmax(dim=-1)
        return {
            "loss_pos": loss,
            "positive_regional_accuracy": predictions.eq(targets).float().mean(),
            "positive_negative_probability": (mass["negative_all"] - denominator).exp().mean(),
        }

    def negative_loss(
        self,
        sim_pos: Tensor,
        sim_neg: Tensor,
        valid_negative: Optional[Tensor],
        logit_scale: Tensor,
    ) -> Dict[str, Tensor]:
        mass = self._log_mass(sim_pos, sim_neg, logit_scale)
        denominator = torch.logaddexp(mass["positive_all"], mass["negative_all"])
        per_sample = -(mass["negative_all"] - denominator).mean(dim=1)
        negative_probability = (mass["negative_all"] - denominator).exp().mean(dim=1)
        if valid_negative is None:
            valid_negative = torch.ones_like(per_sample, dtype=torch.bool)
        valid_negative = valid_negative.to(per_sample.device, dtype=torch.bool)
        if valid_negative.any():
            loss = per_sample[valid_negative].mean()
            probability = negative_probability[valid_negative].mean()
        else:
            # Preserve a valid graph for optimizer/backward when every hard
            # negative was filtered out by crop selection.
            loss = (sim_pos.sum() + sim_neg.sum()) * 0.0
            probability = loss.detach()
        return {
            "loss_neg": loss,
            "negative_prompt_probability": probability,
            "valid_negative_ratio": valid_negative.float().mean(),
        }
