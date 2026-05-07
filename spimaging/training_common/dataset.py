"""PyTorch datasets for generated SPAD .npz samples."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LIGHT_SPEED_M_PER_S = 3e8


def list_sample_files(dataset_dirs):
    files = []
    for dataset_dir in dataset_dirs:
        root = Path(dataset_dir)
        if root.is_file() and root.suffix == ".npz":
            files.append(root)
            continue
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")
        root_files = sorted(root.glob("sample_*.npz"))
        if not root_files:
            root_files = sorted(root.glob("*.npz"))
        files.extend(root_files)
    if not files:
        raise FileNotFoundError(f"No .npz samples found in: {dataset_dirs}")
    return files


def downsample_time(volume, factor):
    """Sum-neighbor downsample a (T,H,W) transient volume along time."""
    factor = int(factor)
    if factor <= 1:
        return volume.astype(np.float32)
    t, h, w = volume.shape
    usable_t = (t // factor) * factor
    if usable_t <= 0:
        raise ValueError(f"temporal_downsample={factor} is too large for T={t}")
    volume = volume[:usable_t]
    return volume.reshape(usable_t // factor, factor, h, w).sum(axis=1).astype(np.float32)


def downsample_space(volume, factor):
    """Sum-neighbor downsample a (T,H,W) transient volume along space."""
    factor = int(factor)
    if factor <= 1:
        return volume.astype(np.float32)
    t, h, w = volume.shape
    usable_h = (h // factor) * factor
    usable_w = (w // factor) * factor
    if usable_h <= 0 or usable_w <= 0:
        raise ValueError(f"spatial_downsample={factor} is too large for H,W={h},{w}")
    volume = volume[:, :usable_h, :usable_w]
    return volume.reshape(t, usable_h // factor, factor, usable_w // factor, factor).sum(axis=(2, 4)).astype(np.float32)


def normalize_counts(counts, use_log=True):
    counts = counts.astype(np.float32)
    if use_log:
        counts = np.log1p(counts)
    scale = float(np.nanmax(counts))
    if scale <= 0:
        scale = 1.0
    counts = counts / scale
    return np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def depth_to_bin(depth_m, bin_size):
    tof_s = 2.0 * depth_m.astype(np.float32) / LIGHT_SPEED_M_PER_S
    return tof_s / float(bin_size)


def gaussian_depth_distribution(depth_m, num_bins, bin_size, temporal_downsample=1, sigma_bins=2.0):
    """Build a per-pixel target distribution over time bins from metric depth."""
    effective_bin_size = float(bin_size) * int(temporal_downsample)
    centers = depth_to_bin(depth_m, effective_bin_size)
    centers = np.clip(centers, 0.0, float(num_bins - 1)).astype(np.float32)
    grid = np.arange(num_bins, dtype=np.float32)[:, None, None]
    sigma = max(float(sigma_bins), 1e-3)
    target = np.exp(-0.5 * ((grid - centers[None, ...]) / sigma) ** 2)
    target_sum = target.sum(axis=0, keepdims=True)
    target = target / np.maximum(target_sum, 1e-8)
    return target.astype(np.float32)


def transient_distribution(transient, eps=1e-8):
    target = np.maximum(transient.astype(np.float32), 0.0)
    target = target / np.maximum(target.sum(axis=0, keepdims=True), eps)
    return target.astype(np.float32)


class SPADHistogramDataset(Dataset):
    """Supervised SPAD histogram dataset for 3D neural networks.

    Input:
        counts as a dense tensor with shape (1,T,H,W).

    Target:
        per-pixel probability distribution over time bins with shape (T,H,W).
        By default this is a Gaussian centered at the ground-truth ToF bin
        derived from depth_m. If target_source="clean" and transient_clean is
        available, the clean transient distribution is used instead.
    """

    def __init__(
        self,
        files,
        temporal_downsample=1,
        target_sigma_bins=2.0,
        target_source="depth",
        use_log_counts=True,
    ):
        self.files = [Path(p) for p in files]
        self.temporal_downsample = int(temporal_downsample)
        self.target_sigma_bins = float(target_sigma_bins)
        self.target_source = str(target_source)
        self.use_log_counts = bool(use_log_counts)

        if self.target_source not in {"depth", "clean"}:
            raise ValueError("target_source must be 'depth' or 'clean'.")

    def __len__(self):
        return len(self.files)

    @property
    def in_channels(self):
        return 1

    def __getitem__(self, index):
        path = self.files[index]
        data = np.load(path, allow_pickle=True)

        counts = downsample_time(data["counts"], self.temporal_downsample)
        counts_input = normalize_counts(counts, use_log=self.use_log_counts)[None, ...]

        depth_m = data["depth_m"].astype(np.float32)
        bin_size = float(data["bin_size"]) if "bin_size" in data else 80e-12

        if self.target_source == "clean" and "transient_clean" in data:
            transient = downsample_time(data["transient_clean"], self.temporal_downsample)
            target = transient_distribution(transient)
        else:
            target = gaussian_depth_distribution(
                depth_m=depth_m,
                num_bins=counts.shape[0],
                bin_size=bin_size,
                temporal_downsample=self.temporal_downsample,
                sigma_bins=self.target_sigma_bins,
            )

        return {
            "input": torch.from_numpy(counts_input),
            "target": torch.from_numpy(target),
            "depth_m": torch.from_numpy(depth_m[None, ...]),
            "bin_size": torch.tensor(bin_size, dtype=torch.float32),
            "path": str(path),
        }


class SPISRSelfSupervisedDataset(Dataset):
    """Raw-measurement dataset for self-supervised SPISR.

    The returned ``measurement`` is a low-resolution photon cube with shape
    (1,Tl,Hl,Wl). If generated high-resolution samples are used for simulation,
    temporal/spatial downsampling creates LR measurements from the stored counts.
    """

    def __init__(
        self,
        files,
        temporal_downsample=1,
        spatial_downsample=1,
        use_log_counts=False,
        normalize=True,
    ):
        self.files = [Path(p) for p in files]
        self.temporal_downsample = int(temporal_downsample)
        self.spatial_downsample = int(spatial_downsample)
        self.use_log_counts = bool(use_log_counts)
        self.normalize = bool(normalize)

    def __len__(self):
        return len(self.files)

    @property
    def in_channels(self):
        return 1

    def __getitem__(self, index):
        path = self.files[index]
        data = np.load(path, allow_pickle=True)
        counts = downsample_time(data["counts"], self.temporal_downsample)
        counts = downsample_space(counts, self.spatial_downsample)
        measurement = counts.astype(np.float32)
        if self.normalize:
            measurement = normalize_counts(measurement, use_log=self.use_log_counts)
        return {
            "measurement": torch.from_numpy(measurement[None, ...]),
            "path": str(path),
        }
