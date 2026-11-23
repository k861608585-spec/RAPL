"""First-stage NR_RAPL model façade.

The class subclasses :class:`RAPL` and intentionally leaves its ``forward``
and ID prediction path untouched.  The negative-region interfaces are present
for the next stage, but no crop-based loss is injected into training yet.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import clip
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys

from RAPL.nr_rapl.config import NRRAPLConfig
from RAPL.vlprompt.rapl import RAPL
from RAPL.vlprompt.tools.regional_contrastive_loss import RegionalContrastiveLoss
from RAPL.vlprompt.tools.topk_reduce import topk_patch_mean


class NRRAPL(RAPL):
    """Backward-compatible RAPL extension with isolated NR parameters."""

    def __init__(self, nr_config: Optional[NRRAPLConfig] = None, **kwargs: Any) -> None:
        # RAPL's class-level list is mutable.  Give this new instance its own
        # list so NR prefixes cannot leak into a later legacy RAPL instance.
        self.TRAINABLE_PARAMS = []
        self.nr_config = nr_config if nr_config is not None else NRRAPLConfig(ENABLED=True)
        super().__init__(**kwargs)
        self.model_name = "NRRAPL_" + self.clip_name[5:]
        if self.nr_config.ABLATION.USE_NR_GLMCM:
            self.ood_method = self.nr_config.OOD_SCORE.TYPE

        guide_prompt = self.global_prompt.detach().clone() if hasattr(self, "global_prompt") else None
        self.register_buffer("frozen_guide_global_prompt", guide_prompt, persistent=True)
        self._guide_loaded_from_checkpoint = False
        self._frozen_guide_text_cache: Optional[Tuple[Tuple[str, ...], Tensor]] = None

        n_ctx = self.nr_config.NEG_PROMPT.N_CTX or self.n_token_context
        self.negative_prompt_learner = nn.Parameter(
            torch.empty(self.nr_config.NEG_PROMPT.NUM, n_ctx, self.ln_final.weight.shape[0])
        )
        nn.init.normal_(self.negative_prompt_learner, std=self.ln_final.weight.shape[0] ** -0.5)
        if self.nr_config.NEG_PROMPT.ENABLED:
            self.TRAINABLE_PARAMS.append("negative_prompt_learner")
        else:
            self.negative_prompt_learner.requires_grad_(False)

        # Checkpoint payload is retained until train.py has constructed its
        # optimizer, scheduler and AMP scaler.
        self._nr_checkpoint_payload: Optional[Dict[str, Any]] = None
        self.regional_criterion = RegionalContrastiveLoss(
            topk=self.topk,
            negative_denom_weight=self.nr_config.OOD_SCORE.NEG_DENOM_WEIGHT,
        )

    @torch.no_grad()
    def initialize_prompt(self) -> None:
        """Initialize RAPL prompts, then initialize the independent negatives."""
        super().initialize_prompt()
        self.initialize_negative_prompts()

    @torch.no_grad()
    def initialize_negative_prompts(self) -> None:
        """Initialize class-agnostic local negative context tokens.

        ``random`` samples context vectors directly; ``local_mean`` starts at
        RAPL's positive local-prompt mean plus small noise; ``text`` copies
        available context-token embeddings from a non-empty neutral template.
        """
        prompts = self.negative_prompt_learner
        init = self.nr_config.NEG_PROMPT.INIT
        if init == "random":
            nn.init.normal_(prompts, std=self.ln_final.weight.shape[0] ** -0.5)
            return
        if init == "local_mean":
            if not hasattr(self, "local_prompts"):
                raise RuntimeError("NEG_PROMPT.INIT=local_mean requires RAPL local prompts")
            source = self.local_prompts.detach().mean(dim=0)
            prompts.zero_()
            copy_tokens = min(source.shape[0], prompts.shape[1])
            prompts[:, :copy_tokens].copy_(source[:copy_tokens].unsqueeze(0).expand(prompts.shape[0], -1, -1))
            prompts.add_(torch.randn_like(prompts) * self.nr_config.NEG_PROMPT.INIT_NOISE_STD)
            return
        if init == "text":
            template = self.nr_config.NEG_PROMPT.TEMPLATE.strip() or "object"
            tokenized = clip.tokenize(template).to(self.device, non_blocking=True)
            embedding = self.token_embedding(tokenized).type_as(prompts)
            prompts.zero_()
            copy_tokens = min(prompts.shape[1], embedding.shape[1] - 2)
            if copy_tokens > 0:
                prompts[:, :copy_tokens].copy_(embedding[:, 1 : 1 + copy_tokens].expand(prompts.shape[0], -1, -1))
            prompts.add_(torch.randn_like(prompts) * self.nr_config.NEG_PROMPT.INIT_NOISE_STD)
            return
        raise ValueError(f"Unsupported NR_RAPL.NEG_PROMPT.INIT: {init}")

    @property
    def uses_guided_crops(self) -> bool:
        return bool(self.nr_config.CROP.ENABLED and self.nr_config.ABLATION.USE_GUIDED_CROP)

    @property
    def uses_regional_contrastive(self) -> bool:
        """Regional training is opt-in and fully disabled with negative prompts."""
        return bool(self.uses_guided_crops and self.nr_config.NEG_PROMPT.ENABLED and self.nr_config.ABLATION.USE_NEG_PROMPT)

    def configure_train_stage(self) -> None:
        """Apply the requested global-prompt training policy after freeze_clip()."""
        stage = self.nr_config.TRAIN_STAGE.lower()
        if stage in {"local", "local_only", "regional"} and hasattr(self, "global_prompt"):
            self.global_prompt.requires_grad_(False)

    def gradient_status(self) -> Dict[str, bool]:
        """Return auditable gradient flags after backward for Stage 2/3."""
        def group_has_grad(prefixes: Tuple[str, ...]) -> bool:
            return any(
                parameter.grad is not None
                for name, parameter in self.named_parameters()
                if name.startswith(prefixes)
            )

        stage2 = self.nr_config.TRAIN_STAGE.lower() in {"local", "local_only", "regional"}
        global_prompt = getattr(self, "global_prompt", None)
        frozen_guide = self.frozen_guide_global_prompt
        return {
            "clip_image_grad_none": not group_has_grad(("visual",)),
            "clip_text_grad_none": not group_has_grad(
                ("token_embedding", "transformer", "ln_final", "text_projection", "positional_embedding", "logit_scale")
            ),
            "guide_global_grad_none": frozen_guide is None or frozen_guide.grad is None,
            "stage2_global_grad_none": (not stage2) or global_prompt is None or global_prompt.grad is None,
            "local_positive_prompt_has_grad": group_has_grad(("local_prompts",)),
            "local_negative_prompt_has_grad": group_has_grad(("negative_prompt_learner",)),
            "local_projection_has_grad": group_has_grad(("local_proj",)),
        }

    def assert_frozen_gradient_invariants(self) -> Dict[str, bool]:
        """Raise if frozen CLIP/guide/global Stage-2 tensors receive gradients."""
        status = self.gradient_status()
        frozen = ("clip_image_grad_none", "clip_text_grad_none", "guide_global_grad_none", "stage2_global_grad_none")
        failures = [name for name in frozen if not status[name]]
        if failures:
            raise RuntimeError(f"Frozen NR_RAPL parameters unexpectedly received gradients: {failures}")
        return status

    def _freeze_clip_encoders_for_nr(self) -> None:
        """Guarantee regional objectives never update CLIP encoder parameters."""
        encoder_prefixes = ("visual", "token_embedding", "transformer", "ln_final", "text_projection", "positional_embedding", "logit_scale")
        for name, parameter in self.named_parameters():
            if name.startswith(encoder_prefixes):
                parameter.requires_grad_(False)

    @torch.no_grad()
    def refresh_frozen_global_guide(self, force: bool = False) -> None:
        """Snapshot the trained RAPL global prompts for crop guidance.

        The snapshot is a non-trainable buffer.  It is deliberately separate
        from ``global_prompt`` so future prompt updates cannot influence crop
        selection within an NR_RAPL training run.
        """
        if self._guide_loaded_from_checkpoint and not force:
            return
        if not hasattr(self, "global_prompt"):
            if self.uses_guided_crops:
                raise RuntimeError("Guided crops require RAPL global prompts")
            return
        self.frozen_guide_global_prompt = self.global_prompt.detach().clone()
        self._frozen_guide_text_cache = None

    @torch.no_grad()
    def encode_frozen_global_guide(self, class_names: List[str]) -> Tensor:
        """Encode frozen global prompts without Prompt Dropout or gradients."""
        if self.frozen_guide_global_prompt is None:
            raise RuntimeError("Frozen global guide has not been initialized")
        signature = tuple(class_names)
        if self._frozen_guide_text_cache is not None:
            cached_signature, cached_features = self._frozen_guide_text_cache
            if cached_signature == signature and cached_features.device == self.device:
                return cached_features
        tokenized_text = clip.tokenize(class_names).to(self.device, non_blocking=True)
        eot_tokens = tokenized_text.argmax(dim=-1)
        token_embeddings = self.token_embedding(tokenized_text)
        n_ctx = self.frozen_guide_global_prompt.shape[1]
        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1:-n_ctx, :]
        text_features = self._most_efficient_encode_text(
            prefix,
            self.frozen_guide_global_prompt.detach(),
            suffix,
            eot_tokens,
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # This cache is safe: the guide is a frozen buffer and crop selection
        # is under no_grad. Learnable local/negative prompt features are never
        # stored across optimizer steps.
        self._frozen_guide_text_cache = (signature, text_features.detach())
        return text_features

    def encode_global_prompts(self, class_names: List[str]) -> Tensor:
        """Return normalized global-prompt features with shape ``[C, G, D]``."""
        global_features, _ = self.encode_text(class_names)
        return global_features / global_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_local_positive_prompts(self, class_names: List[str]) -> Tensor:
        """Return normalized positive local-prompt features ``[C, L, D]``."""
        self._freeze_clip_encoders_for_nr()
        if not (self.learn_local_prompts or self.n_local_prompts > 1):
            _, local_features = self.encode_text(class_names)
            return local_features / local_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        tokenized_text = clip.tokenize(class_names).to(self.device, non_blocking=True)
        eot_tokens = tokenized_text.argmax(dim=-1)
        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)
        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]
        local_features = self._most_efficient_encode_text(prefix, self.local_prompts, suffix, eot_tokens)
        return local_features / local_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_local_negative_prompts(self) -> Optional[Tensor]:
        """Encode class-agnostic negative local prompts as ``[Q, D]``.

        This is intentionally an isolated text-only interface.  Its outputs are
        not copied per class and are not used by RAPL's original ID
        classifier. ``[Q, D]`` differs from positive local prompt features,
        whose RAPL-native order is ``[num_classes, num_local_prompts, D]``.
        """
        if not self.nr_config.NEG_PROMPT.ENABLED or self.nr_config.NEG_PROMPT.NUM < 1:
            return None
        self._freeze_clip_encoders_for_nr()
        n_ctx = self.negative_prompt_learner.shape[1]
        neutral_template = self.nr_config.NEG_PROMPT.TEMPLATE.strip() or "object"
        tokenized_text = clip.tokenize(neutral_template).to(self.device, non_blocking=True)
        eot_tokens = tokenized_text.argmax(dim=-1)
        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)
        prefix = token_embeddings[:, :1, :].expand(self.negative_prompt_learner.size(0), -1, -1)
        suffix = token_embeddings[:, 1:-n_ctx, :].expand(self.negative_prompt_learner.size(0), -1, -1)
        x = torch.cat([prefix, self.negative_prompt_learner, suffix], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x, padding = self.pad_if_necessary(x)
        x, *_ = self.transformer(x, batch_first=True)
        x = self.unpad_if_necessary(x, padding)
        x = self.ln_final(x).type(self.dtype)
        features = x[torch.arange(x.shape[0], device=x.device), eot_tokens.expand(x.shape[0]) + n_ctx] @ self.text_projection
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def select_guided_crops(
        self,
        candidate_crops: Tensor,
        targets: Tensor,
        class_names: List[str],
    ) -> Optional[Dict[str, Tensor]]:
        """Select positive and hard-negative crop indices using frozen global scores.

        ``candidate_crops`` has shape ``[B, Q, 3, H, W]``.  This first-stage
        implementation exposes the selection contract but is not connected to
        the training dataloader/loss yet.
        """
        if not self.training:
            return None
        if candidate_crops.ndim != 5:
            raise ValueError("candidate_crops must have shape [B, Q, 3, H, W]")
        batch, candidates = candidate_crops.shape[:2]
        if targets.shape[0] != batch:
            raise ValueError("targets and candidate_crops must share the batch dimension")
        if candidates < 1:
            raise ValueError("At least one crop candidate is required")

        # This path never calls encode_text(), so RAPL's training-time
        # prompt subsampling/dropout cannot affect guide selection.
        global_text = self.encode_frozen_global_guide(class_names).mean(dim=1)
        flattened = candidate_crops.flatten(0, 1)
        logits_chunks = []
        chunk_size = max(1, self.nr_config.CROP.ENCODE_BATCH_SIZE)
        for start in range(0, flattened.shape[0], chunk_size):
            images_chunk = flattened[start : start + chunk_size].to(self.device, non_blocking=True)
            image_features, _ = self.encode_image_and_proj(images_chunk)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            logits_chunks.append(image_features @ global_text.t())
        logits = torch.cat(logits_chunks, dim=0).view(batch, candidates, -1)
        if logits.shape[-1] < 2:
            raise ValueError("Margin-guided crop selection requires at least two classes")

        targets = targets.to(logits.device, non_blocking=True).long()
        target_logits = logits.gather(2, targets.view(batch, 1, 1).expand(-1, candidates, 1)).squeeze(-1)
        other_logits = logits.clone()
        other_logits.scatter_(2, targets.view(batch, 1, 1).expand(-1, candidates, 1), -torch.inf)
        margin = target_logits - other_logits.max(dim=-1).values
        selection_values = target_logits if self.nr_config.CROP.SELECTION == "similarity" else margin
        positive_indices = selection_values.argmax(dim=1)
        # The prior public fields retain one hard negative. ``*_all`` extends
        # the selection contract for CROP.NUM_NEGATIVE={1,2,4} sweeps without
        # changing the default training behaviour.
        requested_negatives = max(1, self.nr_config.CROP.NUM_NEGATIVE)
        available_negatives = min(requested_negatives, max(0, candidates - 1))
        negative_indices_all = positive_indices.unsqueeze(1).expand(-1, requested_negatives).clone()
        negative_valid_slots = torch.zeros(batch, requested_negatives, dtype=torch.bool, device=logits.device)
        if available_negatives > 0:
            masked_selection = selection_values.clone()
            masked_selection.scatter_(1, positive_indices.unsqueeze(1), torch.inf)
            hard_negative_order = masked_selection.argsort(dim=1)
            selected_negatives = hard_negative_order[:, :available_negatives]
            negative_indices_all[:, :available_negatives] = selected_negatives
            negative_valid_slots[:, :available_negatives] = True
        negative_repeated_all = negative_indices_all.eq(positive_indices.unsqueeze(1))
        negative_margin_all = margin.gather(1, negative_indices_all)
        negative_indices = negative_indices_all[:, 0]
        negative_repeated = negative_repeated_all[:, 0]
        negative_margin = negative_margin_all[:, 0]
        spread = margin.max(dim=1).values - margin.min(dim=1).values
        negative_valid_all = negative_valid_slots & ~negative_repeated_all
        threshold = self.nr_config.CROP.NEGATIVE_MARGIN_THRESHOLD
        if threshold is not None:
            negative_valid_all &= negative_margin_all <= threshold
        negative_valid_all &= spread.unsqueeze(1) > self.nr_config.CROP.MIN_MARGIN_SPREAD
        negative_valid = negative_valid_all[:, 0]

        image_indices = torch.arange(batch, device=candidate_crops.device)
        positive_images = candidate_crops[image_indices, positive_indices.to(candidate_crops.device)]
        negative_images = candidate_crops[image_indices, negative_indices.to(candidate_crops.device)]
        all_image_indices = image_indices.unsqueeze(1).expand(-1, requested_negatives)
        negative_images_all = candidate_crops[all_image_indices, negative_indices_all.to(candidate_crops.device)]
        return {
            "positive_images": positive_images,
            "negative_images": negative_images,
            "positive_indices": positive_indices.detach().cpu(),
            "negative_indices": negative_indices.detach().cpu(),
            "candidate_scores": logits.detach().cpu(),
            "target_scores": target_logits.detach().cpu(),
            "candidate_margins": margin.detach().cpu(),
            "negative_valid": negative_valid.detach().cpu(),
            "negative_repeated": negative_repeated.detach().cpu(),
            "negative_images_all": negative_images_all,
            "negative_indices_all": negative_indices_all.detach().cpu(),
            "negative_valid_all": negative_valid_all.detach().cpu(),
            "negative_repeated_all": negative_repeated_all.detach().cpu(),
        }

    @staticmethod
    def _zero_loss(reference: Optional[Tensor]) -> Tensor:
        if reference is not None:
            return reference.sum() * 0.0
        return torch.zeros((), dtype=torch.float32)

    def encode_local_region_features(self, images: Tensor) -> Tensor:
        """Reuse RAPL's visual local tokens and its sole ``local_proj`` h_theta."""
        self._freeze_clip_encoders_for_nr()
        with torch.no_grad():
            _, local_features = self.encode_image(images.to(self.device, non_blocking=True))
        if local_features is None:
            raise RuntimeError("Regional contrastive training requires use_local_features=True")
        # Keep RAPL's exact projection order: h_theta first, then ViT's
        # frozen visual projection when it exists. No second local mapper.
        local_features = self.local_proj(local_features)
        if hasattr(self.visual, "proj") and self.use_local_features:
            local_features = local_features @ self.visual.proj.detach()
        return local_features / local_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def compute_regional_similarities(
        self,
        images: Tensor,
        class_names: List[str],
        positive_text_features: Optional[Tensor] = None,
        negative_text_features: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """Build positive ``[B,J,L,C]`` and shared-negative ``[B,L,Q]`` scores.

        Optional text features are graph-preserving per-iteration caches. They
        must not be detached because the local positive/negative prompts are
        learnable during Stage 2/3.
        """
        local_features = self.encode_local_region_features(images)
        positive_text = positive_text_features
        if positive_text is None:
            positive_text = self.encode_local_positive_prompts(class_names)  # RAPL order [C, J, D]
        negative_text = negative_text_features
        if negative_text is None and self.nr_config.NEG_PROMPT.ENABLED:
            negative_text = self.encode_local_negative_prompts()
        sim_pos = torch.einsum("bld,cjd->bjlc", local_features, positive_text)
        sim_neg = None if negative_text is None else torch.einsum("bld,qd->blq", local_features, negative_text)
        return {"sim_pos": sim_pos, "sim_neg": sim_neg}

    def compute_positive_regional_loss(
        self,
        sim_pos: Optional[Tensor] = None,
        sim_neg: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        reference: Optional[Tensor] = None,
        **_: Any,
    ) -> Dict[str, Tensor]:
        """Positive-crop class regional loss with shared negative denominator."""
        if sim_pos is None or sim_neg is None or targets is None:
            return {"loss_pos": self._zero_loss(reference), "positive_regional_accuracy": self._zero_loss(reference), "positive_negative_probability": self._zero_loss(reference)}
        return self.regional_criterion.positive_loss(sim_pos, sim_neg, targets, self.logit_scale.exp())

    def compute_negative_regional_loss(
        self,
        sim_pos: Optional[Tensor] = None,
        sim_neg: Optional[Tensor] = None,
        negative_valid: Optional[Tensor] = None,
        reference: Optional[Tensor] = None,
        **_: Any,
    ) -> Dict[str, Tensor]:
        """Hard-negative regional loss; safely returns graph-connected zero if empty."""
        if sim_pos is None or sim_neg is None:
            zero = self._zero_loss(reference)
            return {"loss_neg": zero, "negative_prompt_probability": zero.detach(), "valid_negative_ratio": zero.detach()}
        return self.regional_criterion.negative_loss(sim_pos, sim_neg, negative_valid, self.logit_scale.exp())

    def compute_regional_contrastive_losses(
        self,
        positive_images: Tensor,
        negative_images: Tensor,
        targets: Tensor,
        class_names: List[str],
        negative_valid: Optional[Tensor],
        positive_text_features: Optional[Tensor] = None,
        negative_text_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run regional positive/negative paths using the shared local projection."""
        positive_similarities = self.compute_regional_similarities(
            positive_images, class_names, positive_text_features, negative_text_features
        )
        if positive_similarities["sim_neg"] is None:
            raise RuntimeError("Regional contrastive training requires enabled negative prompts")
        positive = self.compute_positive_regional_loss(
            **positive_similarities,
            targets=targets,
            reference=positive_images,
        )
        if self.nr_config.ABLATION.USE_NEG_LOSS:
            negative_similarities = self.compute_regional_similarities(
                negative_images, class_names, positive_text_features, negative_text_features
            )
            negative = self.compute_negative_regional_loss(
                **negative_similarities,
                negative_valid=negative_valid,
                reference=negative_images,
            )
        else:
            zero = positive["loss_pos"] * 0.0
            negative = {"loss_neg": zero, "negative_prompt_probability": zero.detach(), "valid_negative_ratio": zero.detach()}
        return {**positive, **negative}

    def negative_loss_weight(self, epoch: int) -> float:
        """Linear warmup for the hard-negative regional loss."""
        warmup = self.nr_config.LOSS.NEG_WARMUP_EPOCHS
        if warmup <= 0:
            return self.nr_config.LOSS.LAMBDA_NEG
        return self.nr_config.LOSS.LAMBDA_NEG * min(1.0, float(epoch) / warmup)

    def compute_negative_prompt_diversity(
        self,
        reference: Optional[Tensor] = None,
        negative_text_features: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        """Mean off-diagonal absolute cosine similarity of negative features."""
        if not (self.nr_config.NEG_PROMPT.ENABLED and self.nr_config.ABLATION.USE_DIV_LOSS):
            return self._zero_loss(reference if reference is not None else self.negative_prompt_learner)

        features = negative_text_features if negative_text_features is not None else self.encode_local_negative_prompts()
        assert features is not None
        num_prompts = features.shape[0]
        if num_prompts <= 1:
            return features.sum() * 0.0

        # Float32 accumulation is stable under AMP.  It preserves gradients
        # with respect to the learnable negative context tokens.
        features = features.float()
        chunk_size = self.nr_config.NEG_PROMPT.DIVERSITY_CHUNK_SIZE
        if chunk_size <= 0 or chunk_size >= num_prompts:
            similarity = (features @ features.t()).abs()
            total_similarity = similarity.sum() - similarity.diagonal().sum()
        else:
            total_similarity = features.new_zeros(())
            for start in range(0, num_prompts, chunk_size):
                chunk = features[start : start + chunk_size]
                total_similarity = total_similarity + (chunk @ features.t()).abs().sum()
            total_similarity = total_similarity - (features * features).sum(dim=-1).abs().sum()
        return total_similarity / (num_prompts * (num_prompts - 1))

    @torch.no_grad()
    def _normalized_ood_score_type(self, score_type: Optional[str] = None) -> str:
        score_type = self.nr_config.OOD_SCORE.TYPE if score_type is None else score_type
        return score_type.upper().replace("-", "_")

    @property
    def uses_id_confidence_ood_scorer(self) -> bool:
        return bool(
            self.nr_config.ABLATION.USE_NR_GLMCM
            and self._normalized_ood_score_type() in {"REGIONAL_MCM_NO_NEG", "NR_GLMCM"}
        )

    @torch.no_grad()
    def score_ood_images(
        self,
        images: Tensor,
        class_names: List[str],
        global_logits: Tensor,
        local_logits: Optional[Tensor] = None,
        score_type: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """Return high-ID-confidence OOD scoring details without changing ID logits.

        This method is evaluation-only and never invokes random crop selection.
        ``predicted_class`` is derived from RAPL's unchanged positive-prompt
        classifier; negative prompts only enter the local OOD denominator.
        """
        score_type = self._normalized_ood_score_type(score_type)
        global_class_logits = global_logits.mean(dim=-1)
        global_confidence = torch.softmax(global_class_logits / self.ood_temp_scale, dim=-1).max(dim=-1).values

        if local_logits is not None:
            gl_probs, _, _ = self.create_prediction_scores(global_logits, local_logits)
            predicted_class = gl_probs.argmax(dim=-1)
        else:
            predicted_class = global_class_logits.argmax(dim=-1)

        if score_type == "MCM":
            zero = global_confidence.new_zeros((global_confidence.shape[0], 1))
            return {
                "global_confidence": global_confidence,
                "local_confidence_per_scale": zero,
                "local_confidence_mean": zero.squeeze(1),
                "negative_probability": zero.squeeze(1),
                "final_ood_score": global_confidence,
                "predicted_class": predicted_class,
            }

        if score_type == "GL_MCM" and local_logits is not None:
            # Compatibility path: original positive local logits, no negatives.
            positive_logits = (local_logits.permute(0, 3, 1, 2) / self.ood_temp_scale).float()
        else:
            similarities = self.compute_regional_similarities(images, class_names)
            positive_logits = (similarities["sim_pos"] / self.ood_temp_scale).float()

        # [B, J, L, C] -> log P_pos.  Negatives are deliberately absent from
        # REGIONAL_MCM_NO_NEG and all compatibility paths.
        log_positive_all = torch.logsumexp(positive_logits, dim=-1)
        # A disabled negative-prompt ablation must never leak negatives into
        # the regional denominator, even when a checkpoint still contains
        # their parameters.
        negatives_enabled = self.nr_config.NEG_PROMPT.ENABLED and self.nr_config.ABLATION.USE_NEG_PROMPT
        sim_neg = (
            similarities["sim_neg"]
            if score_type not in {"GL_MCM"} and "similarities" in locals() and negatives_enabled
            else None
        )
        if score_type == "NR_GLMCM" and sim_neg is not None:
            negative_logits = (sim_neg / self.ood_temp_scale).float().unsqueeze(1)
            log_negative_all = torch.logsumexp(negative_logits, dim=-1)
            weight = self.nr_config.OOD_SCORE.NEG_DENOM_WEIGHT
            if weight > 0:
                log_denominator = torch.logaddexp(
                    log_positive_all,
                    log_negative_all + torch.tensor(weight, dtype=positive_logits.dtype, device=positive_logits.device).log(),
                )
                negative_probability = (log_negative_all - log_denominator).exp().mean(dim=(1, 2))
            else:
                log_denominator = log_positive_all
                negative_probability = global_confidence.new_zeros(global_confidence.shape)
        else:
            log_denominator = log_positive_all
            negative_probability = global_confidence.new_zeros(global_confidence.shape)

        patch_confidence = (positive_logits.max(dim=-1).values - log_denominator).exp()
        test_topk = self.nr_config.OOD_SCORE.TEST_TOPK or self.topk
        local_confidence_per_scale = topk_patch_mean(patch_confidence, test_topk)
        local_confidence_mean = local_confidence_per_scale.mean(dim=-1)
        final_score = global_confidence + self.nr_config.OOD_SCORE.LOCAL_WEIGHT * local_confidence_mean
        return {
            "global_confidence": global_confidence,
            "local_confidence_per_scale": local_confidence_per_scale,
            "local_confidence_mean": local_confidence_mean,
            "negative_probability": negative_probability,
            "final_ood_score": final_score,
            "predicted_class": predicted_class,
        }

    @torch.no_grad()
    def score_ood_logits(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
        negative_local_logits: Optional[Tensor] = None,
        score_type: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """Score precomputed logits; useful for compatibility and unit tests.

        Negative local logits are optional.  Without them this is the stable
        ``REGIONAL_MCM_NO_NEG`` variant and remains usable with a RAPL
        checkpoint that has no negative prompt learner.
        """
        score_type = self._normalized_ood_score_type(score_type)
        global_class_logits = global_logits.mean(dim=-1)
        global_confidence = torch.softmax(global_class_logits / self.ood_temp_scale, dim=-1).max(dim=-1).values
        if local_logits is None or score_type == "MCM":
            zero = global_confidence.new_zeros((global_confidence.shape[0], 1))
            return {
                "global_confidence": global_confidence,
                "local_confidence_per_scale": zero,
                "local_confidence_mean": zero.squeeze(1),
                "negative_probability": zero.squeeze(1),
                "final_ood_score": global_confidence,
                "predicted_class": global_class_logits.argmax(dim=-1),
            }

        positive_logits = (local_logits.permute(0, 3, 1, 2) / self.ood_temp_scale).float()
        log_positive_all = torch.logsumexp(positive_logits, dim=-1)
        use_negatives = (
            score_type == "NR_GLMCM"
            and negative_local_logits is not None
            and self.nr_config.NEG_PROMPT.ENABLED
            and self.nr_config.ABLATION.USE_NEG_PROMPT
        )
        if use_negatives and self.nr_config.OOD_SCORE.NEG_DENOM_WEIGHT > 0:
            log_negative_all = torch.logsumexp(
                (negative_local_logits / self.ood_temp_scale).float().unsqueeze(1), dim=-1
            )
            log_denominator = torch.logaddexp(
                log_positive_all,
                log_negative_all + torch.tensor(
                    self.nr_config.OOD_SCORE.NEG_DENOM_WEIGHT,
                    device=positive_logits.device,
                    dtype=positive_logits.dtype,
                ).log(),
            )
            negative_probability = (log_negative_all - log_denominator).exp().mean(dim=(1, 2))
        else:
            log_denominator = log_positive_all
            negative_probability = global_confidence.new_zeros(global_confidence.shape)
        patch_confidence = (positive_logits.max(dim=-1).values - log_denominator).exp()
        local_confidence_per_scale = topk_patch_mean(
            patch_confidence,
            self.nr_config.OOD_SCORE.TEST_TOPK or self.topk,
        )
        local_confidence_mean = local_confidence_per_scale.mean(dim=-1)
        final_score = global_confidence + self.nr_config.OOD_SCORE.LOCAL_WEIGHT * local_confidence_mean
        gl_probs, _, _ = self.create_prediction_scores(global_logits, local_logits)
        return {
            "global_confidence": global_confidence,
            "local_confidence_per_scale": local_confidence_per_scale,
            "local_confidence_mean": local_confidence_mean,
            "negative_probability": negative_probability,
            "final_ood_score": final_score,
            "predicted_class": gl_probs.argmax(dim=-1),
        }

    @torch.no_grad()
    def compute_nr_glmcm_score(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
        negative_local_logits: Optional[Tensor] = None,
    ) -> np.ndarray:
        """Return OOD-positive evidence for legacy metric callers.

        The public detail APIs return high ID confidence; the original RAPL
        metric convention is the negative of that value (OOD high).
        """
        details = self.score_ood_logits(global_logits, local_logits, negative_local_logits, "NR_GLMCM")
        return (-details["final_ood_score"]).cpu().numpy()

    def compute_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
        ood_method: Optional[str] = None,
    ) -> np.ndarray:
        ood_method = self.ood_method if ood_method is None else ood_method
        normalized_method = ood_method.upper().replace("-", "_")
        if normalized_method == "GL_MCM":
            return super().compute_scores(global_logits, local_logits, "GL-MCM")
        if normalized_method == "MCM":
            return super().compute_scores(global_logits, local_logits, "MCM")
        if normalized_method == "REGIONAL_MCM_NO_NEG":
            return (-self.score_ood_logits(global_logits, local_logits, score_type="REGIONAL_MCM_NO_NEG")["final_ood_score"]).cpu().numpy()
        if normalized_method == "NR_GLMCM" and self.nr_config.ABLATION.USE_NR_GLMCM:
            return self.compute_nr_glmcm_score(global_logits, local_logits)
        return super().compute_scores(global_logits, local_logits, ood_method)

    def trainable_state_dict(self) -> Dict[str, Any]:
        """Save NR parameters in isolated groups without changing legacy files."""
        state = self.state_dict()
        return {
            "global_prompt_learner": {k: state[k] for k in state if k.startswith("global_prompt")},
            "local_prompt_learner": {k: state[k] for k in state if k.startswith("local_prompt")},
            "local_projection": {k: state[k] for k in state if k.startswith("local_proj")},
            "negative_prompt_learner": {"negative_prompt_learner": state["negative_prompt_learner"]},
            "guide_global_prompt": {"frozen_guide_global_prompt": state["frozen_guide_global_prompt"]}
            if self.frozen_guide_global_prompt is not None else {},
            "NR_RAPL_metadata": self.nr_config.to_dict(),
        }

    def load_trainable_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> _IncompatibleKeys:
        """Accept both legacy flat RAPL checkpoints and NR grouped payloads."""
        group_names = {
            "global_prompt_learner",
            "local_prompt_learner",
            "local_projection",
            "negative_prompt_learner",
            "guide_global_prompt",
        }
        if any(key in state_dict for key in group_names):
            flat_state: Dict[str, Tensor] = {}
            for group_name in group_names:
                values = state_dict.get(group_name, {})
                if isinstance(values, dict):
                    flat_state.update(values)
            keys = self.load_state_dict(flat_state, strict=False)
            self._guide_loaded_from_checkpoint = "guide_global_prompt" in state_dict
            self._frozen_guide_text_cache = None
        else:
            # Original RAPL checkpoints store a flat trainable state dict.
            keys = self.load_state_dict(state_dict, strict=False)

        missing = [
            key for key in keys.missing_keys
            if any(key.startswith(prefix) for prefix in self.TRAINABLE_PARAMS)
            and not key.startswith("negative_prompt_learner")
        ]
        if strict and (keys.unexpected_keys or missing):
            raise RuntimeError(
                "Error loading NR_RAPL checkpoint: "
                f"missing={missing}, unexpected={keys.unexpected_keys}"
            )
        return _IncompatibleKeys(missing_keys=missing, unexpected_keys=keys.unexpected_keys)

    def load_checkpoint_payload(self, payload: Dict[str, Any], strict: bool = False) -> _IncompatibleKeys:
        """Load model parameters now and retain optimiser state for restoration."""
        self._nr_checkpoint_payload = payload
        return self.load_trainable_state_dict(payload["state_dict"], strict=strict)

    def restore_training_state(self, optimizer: Any, scheduler: Any, scaler: Any) -> Optional[int]:
        """Restore optional train-state entries after their objects are built."""
        if self._nr_checkpoint_payload is None:
            return None
        payload = self._nr_checkpoint_payload
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        return payload.get("epoch")
