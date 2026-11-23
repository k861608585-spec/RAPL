from typing import Type, Union
import os
from argparse import Namespace

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from RAPL.lib import DictAverage

NoneType = Type[None]


def save_checkpoint(
    save_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    scaler: Union[GradScaler, NoneType],
    train_meter: DictAverage,
    args: Namespace,
    best_metric: float = None,
    checkpoint_name: str = None,
) -> NoneType:
    state = {
        "epoch": epoch,
        "train_meter": {k: v.avg for k, v in train_meter.items()},
        "state_dict": model.state_dict() if not hasattr(model, "trainable_state_dict") else model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "args": args,
        # Keep a self-contained, serialisable snapshot for both the legacy
        # RAPL stage and NR_RAPL stages.  ``args`` is preserved below for
        # backward compatibility with existing checkpoints.
        "config": (
            model.nr_config.to_dict()
            if hasattr(model, "nr_config") and hasattr(model.nr_config, "to_dict")
            else vars(args).copy()
        ),
        "best_metric": best_metric,
    }

    model_type = model.model_name.split("_")[0]
    dataset_name = args.dataset_name
    archi = "_".join(model.model_name.split("_")[1:])

    os.makedirs(save_dir, exist_ok=True)

    # Legacy filename remains unchanged unless a stage script requests a name.
    name = "_".join([model_type, dataset_name, archi])
    model_name = checkpoint_name or (f"{name}_seed{args.seed}.ckpt" if args.seed is not None else "{name}.ckpt")
    torch.save(state, os.path.join(save_dir, model_name))
