"""Fail-fast dependency and CUDA verification for NR_RAPL."""

from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = ("torch", "torchvision", "clip", "timm", "numpy", "pandas", "sklearn", "rich", "colorlog", "nltk")


def main() -> None:
    missing = []
    modules = {}
    for module_name in REQUIRED_MODULES:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError as error:
            missing.append(f"{module_name}: {error}")
    if missing:
        raise RuntimeError("Missing NR_RAPL dependencies:\n  " + "\n  ".join(missing))

    torch = modules["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. NR_RAPL training is CUDA-only; verify the NVIDIA driver, "
            "CUDA-visible GPU, and the PyTorch CUDA build."
        )

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {modules['torchvision'].__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("NR_RAPL environment verification passed.")


if __name__ == "__main__":
    main()
