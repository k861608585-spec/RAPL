"""CPU-side random crop candidates for the optional NR_RAPL training path."""

import os
from typing import Any, Dict, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from RAPL.nr_rapl.config import CropConfig


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def make_guided_crop_candidates(
    path: Union[str, os.PathLike],
    crop_config: CropConfig,
    image_size: int = 224,
) -> Tensor:
    """Create normalized candidates for one image on CPU.

    This is shared by the dataset wrapper and the training-loop compatibility
    fallback.  The returned tensor has shape ``[M, 3, H, W]``.
    """
    if crop_config.NUM_CANDIDATES < 1:
        raise ValueError("Guided crop generation requires NUM_CANDIDATES >= 1")
    if image_size < 1:
        raise ValueError("Guided crop generation requires image_size >= 1")
    if not (0.0 < crop_config.MIN_SCALE <= crop_config.MAX_SCALE <= 1.0):
        raise ValueError("Crop scales must satisfy 0 < MIN_SCALE <= MAX_SCALE <= 1")

    with Image.open(path) as image:
        image = image.convert("RGB")
        candidates = []
        for _ in range(crop_config.NUM_CANDIDATES):
            top, left, height, width = RandomResizedCrop.get_params(
                image,
                scale=(crop_config.MIN_SCALE, crop_config.MAX_SCALE),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            )
            crop = F.resized_crop(
                image,
                top,
                left,
                height,
                width,
                [image_size, image_size],
                interpolation=InterpolationMode.BICUBIC,
            )
            if torch.rand(()) < 0.5:
                crop = F.hflip(crop)
            candidates.append(F.normalize(F.to_tensor(crop), mean=CLIP_MEAN, std=CLIP_STD))

    # Default DataLoader collation turns this into [B, M, 3, H, W].
    return torch.stack(candidates, dim=0).contiguous()


class GuidedCropDataset(Dataset):
    """Wrap a training dataset and attach normalized random crop candidates.

    The original sample remains untouched under ``image``. Candidates are built
    from the image path on CPU and therefore are never kept as a ``B x M`` GPU
    tensor. The wrapper is intentionally used only for the training dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        crop_config: CropConfig,
        image_size: int = 224,
        enabled: bool = True,
    ) -> None:
        if crop_config.NUM_CANDIDATES < 1:
            raise ValueError("GuidedCropDataset requires NUM_CANDIDATES >= 1")
        if image_size < 1:
            raise ValueError("GuidedCropDataset requires image_size >= 1")
        if not (0.0 < crop_config.MIN_SCALE <= crop_config.MAX_SCALE <= 1.0):
            raise ValueError("Crop scales must satisfy 0 < MIN_SCALE <= MAX_SCALE <= 1")
        self.dataset = dataset
        self.crop_config = crop_config
        self.image_size = image_size
        self.enabled = enabled

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        if name in {"dataset", "crop_config", "image_size", "enabled"}:
            return super().__getattribute__(name)
        return getattr(self.dataset, name)

    def _make_candidates(self, path: Union[str, os.PathLike]) -> Tensor:
        return make_guided_crop_candidates(path, self.crop_config, self.image_size)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        out = self.dataset[index]
        if self.enabled:
            if not isinstance(out, dict):
                raise TypeError(
                    "GuidedCropDataset requires the wrapped dataset to return "
                    f"a dict, got {type(out).__name__} at index {index}"
                )
            if "path" not in out:
                raise KeyError(
                    "GuidedCropDataset requires the wrapped dataset to return "
                    f"a 'path' field (missing at index {index})"
                )
            out = dict(out)
            out["guided_crops"] = self._make_candidates(out["path"])
        return out


def maybe_wrap_guided_crop_dataset(dataset: Dataset, crop_config: CropConfig, use_guided_crop: bool) -> Dataset:
    """Return the original dataset when guided crops are disabled."""
    if not (use_guided_crop and crop_config.ENABLED):
        return dataset
    return GuidedCropDataset(dataset, crop_config)
