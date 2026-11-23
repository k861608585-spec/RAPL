from typing import List, Optional, Type

import torch
from torch import Tensor

NoneType = Type[None]


def _validate_topk(topk: List[int], n_patches: int, n_prompts: Optional[int] = None) -> None:
    """Reject invalid reductions before PyTorch raises an opaque topk error."""
    if not topk:
        raise ValueError("topk must contain at least one positive integer")
    if any(k <= 0 for k in topk):
        raise ValueError(f"topk values must be positive, got {topk}")
    if n_patches <= 0:
        raise ValueError("Local feature extraction produced zero patch tokens; Top-K reduction is undefined")
    if n_prompts is not None and len(topk) not in (1, n_prompts):
        raise ValueError(f"Provide one k or one k per local prompt (got {len(topk)} k values for {n_prompts} prompts)")


def topk_local_positions(local_scores: Tensor, topk: List[int]) -> Tensor:
    """Select RAPL Top-K local positions without reducing class/prompt axes.

    ``local_scores`` follows RAPL's native ``[B, L, C, J]`` layout.  Top-K
    is always applied only along local-token axis ``L`` and is shared by the
    original local classifier and NR regional losses.
    """
    if local_scores.ndim != 4:
        raise ValueError("local_scores must have shape [B, L, C, J]")
    _, n_patches, _, n_prompts = local_scores.shape
    _validate_topk(topk, n_patches, n_prompts)
    maxk = min(max(topk), n_patches)
    return local_scores.topk(dim=1, k=maxk).values


def topk_logsumexp(local_scores: Tensor, topk: List[int]) -> Tensor:
    """Stable log TopKSumExp with output ``[B, J, C]``.

    This preserves RAPL's per-prompt local-position Top-K selection while
    avoiding an explicit exponentiated tensor for regional contrastive losses.
    """
    values = topk_local_positions(local_scores, topk)
    if len(topk) == 1:
        return torch.logsumexp(values[:, : min(topk[0], values.shape[1])], dim=1).permute(0, 2, 1)
    return torch.stack(
        [torch.logsumexp(values[:, : min(k, values.shape[1]), :, prompt], dim=1) for prompt, k in enumerate(topk)],
        dim=1,
    )


def topk_logsumexp_shared(local_scores: Tensor, topk: List[int], n_prompts: Optional[int] = None) -> Tensor:
    """Stable log TopKSumExp for shared negatives, output ``[B, J, Q]``.

    ``local_scores`` is ``[B, L, Q]``.  Negative text features are shared
    across local scales and therefore encoded once, while each scale still uses
    its own RAPL ``k_j`` during the positional reduction.
    """
    if local_scores.ndim != 3:
        raise ValueError("local_scores must have shape [B, L, Q]")
    _validate_topk(topk, local_scores.shape[1])
    maxk = min(max(topk), local_scores.shape[1])
    values = local_scores.topk(dim=1, k=maxk).values
    if len(topk) == 1:
        reduced = torch.logsumexp(values[:, : min(topk[0], values.shape[1])], dim=1).unsqueeze(1)
        return reduced.expand(-1, n_prompts or 1, -1)
    k_values = topk
    return torch.stack(
        [torch.logsumexp(values[:, : min(k, values.shape[1])], dim=1) for k in k_values],
        dim=1,
    )


def topk_patch_mean(patch_scores: Tensor, topk: List[int]) -> Tensor:
    """Mean top local confidences with output ``[B, J]``.

    ``patch_scores`` is ``[B, J, L]``. A single ``k`` is broadcast over all
    scales, while a list uses the RAPL scale order. Values are clamped to the
    available token count for variable test resolutions.
    """
    if patch_scores.ndim != 3:
        raise ValueError("patch_scores must have shape [B, J, L]")
    _, prompts, patches = patch_scores.shape
    _validate_topk(topk, patches, prompts)
    if len(topk) == 1:
        k = min(topk[0], patches)
        return patch_scores.topk(k, dim=-1).values.mean(dim=-1)
    return torch.stack(
        [patch_scores[:, prompt].topk(min(k, patches), dim=-1).values.mean(dim=-1) for prompt, k in enumerate(topk)],
        dim=1,
    )


def topk_reduce(
    local_logits: Tensor,
    topk: Optional[List[int]] = None,
) -> Tensor:
    """
    local_logits is a Tensor of shape (b, p, k, 1) or (b, p, k, m)
    """
    if topk is None:
        return local_logits

    _, _, _, n_prompts = local_logits.shape
    local_logits = topk_local_positions(local_logits, topk)

    if len(topk) == 1:
        local_logits = local_logits.mean(dim=1)
    else:
        local_logits = torch.stack([local_logits[:, :k, :, i].mean(dim=1) for i, k in enumerate(topk)], dim=-1)

    return local_logits
