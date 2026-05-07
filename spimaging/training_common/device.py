"""Device selection helpers."""

from __future__ import annotations


def get_torch_device(prefer_gpu: bool = True):
    """Return CUDA when available, otherwise CPU."""
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
