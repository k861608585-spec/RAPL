from typing import Type
import torch
import RAPL.lib as lib

NoneType = Type[None]


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> NoneType:
    if (checkpoint_path is not None) and (checkpoint_path.lower() != "none"):
        lib.LOGGER.info(f"Loading checkpoint {checkpoint_path}")
        # Production runs retain CUDA loading, while the CPU mock smoke test
        # can validate the same grouped/legacy checkpoint compatibility.
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        # The legacy branch is intentionally unchanged.  NR_RAPL owns the
        # grouped checkpoint format and optional optimiser-state restoration.
        if hasattr(model, "load_checkpoint_payload"):
            keys = model.load_checkpoint_payload(state_dict, strict=False)
        else:
            keys = model.load_trainable_state_dict(state_dict["state_dict"], strict=False)
        if len(keys.unexpected_keys) > 0:
            raise RuntimeError(f"Unexpected keys in state_dict: {keys.unexpected_keys}")
        if len(keys.missing_keys) > 0:
            lib.LOGGER.warning(f"Missing keys in state_dict: {keys.missing_keys}")
