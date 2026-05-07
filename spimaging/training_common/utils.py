"""Shared training utilities."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F


def split_indices(n_samples, val_fraction, seed):
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)
    n_val = int(round(n_samples * val_fraction))
    if n_samples > 1:
        n_val = min(max(n_val, 1), n_samples - 1)
    else:
        n_val = 0
    return indices[n_val:], indices[:n_val]


def match_distribution_shape(logits, target_distribution):
    """Resize logits to match a target distribution shaped (B,T,H,W)."""
    if logits.shape[2:] == target_distribution.shape[1:]:
        return logits
    return F.interpolate(logits, size=target_distribution.shape[1:], mode="trilinear", align_corners=False)


def match_volume_shape(volume, reference):
    """Resize a 5D volume to match another 5D volume's (T,H,W)."""
    if volume.shape[2:] == reference.shape[2:]:
        return volume
    return F.interpolate(volume, size=reference.shape[2:], mode="trilinear", align_corners=False)


def save_training_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    args,
    model_name,
    method_family,
    best_metric,
    best_metric_name,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "model_name": model_name,
            "method_family": method_family,
            best_metric_name: best_metric,
        },
        path,
    )
