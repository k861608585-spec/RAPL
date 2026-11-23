"""Explicit optimizer parameter groups for NR_RAPL stages 2 and 3."""

from typing import Any, Dict, List

from torch import nn
from torch.optim import Adam, AdamW, Optimizer, SGD

import RAPL.lib as lib


def NR_RAPL_parameter_groups(model: nn.Module, learning_rate: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Return only trainable NR prompt/projection groups, never CLIP weights."""
    config = model.nr_config.OPTIM
    definitions = [
        ("global_prompt_params", "global_prompt", config.GLOBAL_LR_MULTIPLIER),
        ("local_positive_prompt_params", "local_prompts", config.LOCAL_POSITIVE_LR_MULTIPLIER),
        ("local_negative_prompt_params", "negative_prompt_learner", config.LOCAL_NEGATIVE_LR_MULTIPLIER),
        ("local_projection_params", "local_proj", config.LOCAL_PROJECTION_LR_MULTIPLIER),
    ]
    groups: List[Dict[str, Any]] = []
    assigned = set()
    for group_name, prefix, multiplier in definitions:
        parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix) and parameter.requires_grad]
        assigned.update(id(parameter) for parameter in parameters)
        groups.append(
            {
                "name": group_name,
                "params": parameters,
                "lr": learning_rate * multiplier,
                "weight_decay": weight_decay,
            }
        )
        lib.LOGGER.info(
            f"{group_name}: lr={learning_rate * multiplier:g}, weight_decay={weight_decay:g}, "
            f"requires_grad={bool(parameters)}, params={len(parameters)}"
        )

    unexpected = [name for name, parameter in model.named_parameters() if parameter.requires_grad and id(parameter) not in assigned]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters outside NR_RAPL groups: {unexpected}")
    return [group for group in groups if group["params"]]


def get_NR_RAPL_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> Optimizer:
    groups = NR_RAPL_parameter_groups(model, learning_rate, weight_decay)
    if not groups:
        raise RuntimeError("NR_RAPL has no trainable parameter groups for the selected stage")
    if optimizer_name == "adam":
        return Adam(groups)
    if optimizer_name == "adamw":
        return AdamW(groups, eps=1e-4)
    if optimizer_name == "sgd":
        return SGD(groups, momentum=momentum)
    raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")
