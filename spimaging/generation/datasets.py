"""Dataset scanning and loading utilities for SPAD generation."""

from pathlib import Path
import re

import imageio.v2 as imageio
import numpy as np


# =========================================================
# NYUv2 labeled
# =========================================================
def load_nyuv2_labeled(nyu_mat_path: Path):
    import h5py

    with h5py.File(nyu_mat_path, "r") as f:
        images = np.array(f["images"])
        depths = np.array(f["depths"])

    if images.ndim != 4 or depths.ndim != 3:
        raise ValueError(f"Unexpected NYUv2 shapes: images={images.shape}, depths={depths.shape}")

    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    elif images.shape[2] == 3:
        images = np.transpose(images, (3, 0, 1, 2))
    elif images.shape[3] == 3:
        pass
    else:
        raise ValueError(f"Cannot infer image layout from shape {images.shape}")

    if depths.shape[0] == images.shape[0]:
        pass
    elif depths.shape[2] == images.shape[0]:
        depths = np.transpose(depths, (2, 0, 1))
    else:
        raise ValueError(f"Cannot infer depth layout from shape {depths.shape}")

    return images.astype(np.float32), depths.astype(np.float32)


# =========================================================
# NYUv2 raw
# =========================================================
_TIMESTAMP_RE = re.compile(r"^[ard]-(\d+\.\d+)-\d+\.(ppm|pgm|dump)$")


def parse_nyu_timestamp(path: Path) -> float:
    m = _TIMESTAMP_RE.match(path.name)
    if m is None:
        raise ValueError(f"Unexpected NYUv2 raw filename: {path.name}")
    return float(m.group(1))


def load_raw_pair(rgb_path: Path, depth_path: Path, max_valid_depth_m: float = 10.0):
    rgb = imageio.imread(rgb_path)
    depth = imageio.imread(depth_path)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Unexpected RGB shape: {rgb.shape} from {rgb_path}")
    if depth.ndim != 2:
        raise ValueError(f"Unexpected depth shape: {depth.shape} from {depth_path}")

    rgb = rgb.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32)

    valid = np.isfinite(depth) & (depth > 0.05) & (depth <= float(max_valid_depth_m))
    if np.any(valid):
        fill_value = np.median(depth[valid]).astype(np.float32)
    else:
        fill_value = np.float32(1.0)
    depth = np.where(valid, depth, fill_value).astype(np.float32)
    return rgb, depth


def scan_nyuv2_raw_pairs(raw_root: Path, max_time_diff: float = 0.05, stride: int = 1, drop_first: int = 5, drop_last: int = 5, verbose: bool = True):
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root not found: {raw_root}")

    scene_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    all_pairs = []

    for scene_dir in scene_dirs:
        rgb_files = sorted(scene_dir.glob("r-*.ppm"))
        depth_files = sorted(scene_dir.glob("d-*.pgm"))
        if len(rgb_files) == 0 or len(depth_files) == 0:
            continue

        rgb_ts = np.array([parse_nyu_timestamp(p) for p in rgb_files], dtype=np.float64)
        depth_ts = np.array([parse_nyu_timestamp(p) for p in depth_files], dtype=np.float64)

        scene_pairs = []
        for i, d_path in enumerate(depth_files):
            d_t = depth_ts[i]
            j = int(np.argmin(np.abs(rgb_ts - d_t)))
            dt = abs(rgb_ts[j] - d_t)
            if dt <= max_time_diff:
                scene_pairs.append(
                    {
                        "scene": scene_dir.name,
                        "rgb_path": rgb_files[j],
                        "depth_path": d_path,
                        "rgb_ts": float(rgb_ts[j]),
                        "depth_ts": float(d_t),
                        "dt": float(dt),
                    }
                )

        unique = {}
        for item in scene_pairs:
            key = str(item["depth_path"])
            if key not in unique or item["dt"] < unique[key]["dt"]:
                unique[key] = item

        scene_pairs = list(unique.values())
        scene_pairs = sorted(scene_pairs, key=lambda x: x["depth_ts"])
        original_len = len(scene_pairs)

        if len(scene_pairs) > (drop_first + drop_last):
            scene_pairs = scene_pairs[drop_first: len(scene_pairs) - drop_last]
        else:
            scene_pairs = []

        if stride > 1:
            scene_pairs = scene_pairs[::stride]

        if verbose:
            print(f"{scene_dir.name}: matched={original_len}, kept_after_trim={len(scene_pairs)}")

        all_pairs.extend(scene_pairs)

    return all_pairs


# =========================================================
# Middlebury
# =========================================================
def scan_middlebury_pairs(middlebury_root: Path, verbose: bool = True):
    if not middlebury_root.exists():
        raise FileNotFoundError(f"Middlebury root not found: {middlebury_root}")

    scene_dirs = sorted([p for p in middlebury_root.iterdir() if p.is_dir()])
    pairs = []

    for scene_dir in scene_dirs:
        rgb_path = scene_dir / "view1.png"
        disp_path = scene_dir / "disp1.png"
        dmin_path = scene_dir / "dmin.txt"

        if rgb_path.exists() and disp_path.exists():
            pairs.append(
                {
                    "scene": scene_dir.name,
                    "rgb_path": rgb_path,
                    "disp_path": disp_path,
                    "dmin_path": dmin_path if dmin_path.exists() else None,
                }
            )
            if verbose:
                print(f"{scene_dir.name}: found view1.png + disp1.png")
        else:
            if verbose:
                print(f"{scene_dir.name}: skipped (missing view1.png or disp1.png)")

    return pairs


def load_middlebury_pair(rgb_path: Path, disp_path: Path, depth_min=0.5, depth_max=5.0):
    rgb = imageio.imread(rgb_path).astype(np.float32)
    disp = imageio.imread(disp_path).astype(np.float32)

    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.ndim == 3 and rgb.shape[2] > 3:
        rgb = rgb[..., :3]

    if disp.ndim == 3:
        disp = disp[..., 0]

    disp[~np.isfinite(disp)] = 0.0
    disp[disp < 0.0] = 0.0

    valid = disp > 0
    if valid.sum() == 0:
        raise ValueError(f"No valid disparity values found in {disp_path}")

    eps = 1e-6
    z_rel = np.zeros_like(disp, dtype=np.float32)
    z_rel[valid] = 1.0 / (disp[valid] + eps)

    z_min = float(z_rel[valid].min())
    z_max = float(z_rel[valid].max())

    depth = np.zeros_like(z_rel, dtype=np.float32)
    if z_max > z_min:
        depth[valid] = depth_min + (z_rel[valid] - z_min) / (z_max - z_min) * (depth_max - depth_min)
    else:
        depth[valid] = (depth_min + depth_max) / 2.0

    median_depth = np.median(depth[valid]).astype(np.float32)
    depth[~valid] = median_depth
    return rgb.astype(np.float32), depth.astype(np.float32)
