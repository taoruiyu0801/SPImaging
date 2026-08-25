"""Build the redistributable SPImaging synthetic public demo assets.

The generator deliberately has no input-data argument: every RGB image, depth
map and SPAD measurement is produced from analytic geometry in this file.  This
prevents release builds from accidentally depending on the private/derived demo
data used elsewhere in the repository.

The generated arrays are dedicated to the public domain under CC0-1.0.  The
generator source itself is covered by the repository's Apache-2.0 license.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Callable
import zipfile

import numpy as np


SCHEMA_VERSION = 1
GENERATOR_VERSION = "1.0.0"
HEIGHT = 64
WIDTH = 64
TEMPORAL_BINS = 1024
BIN_SIZE_SECONDS = 80e-12
LIGHT_SPEED_M_PER_S = 3e8
MEAN_SIGNAL_PHOTONS = 16.0
MEAN_BACKGROUND_PHOTONS = 4.0
PULSE_SIGMA_BINS = 2.0
BASE_SEED = 2026082301
SAMPLE_NAMES = tuple(f"sample_{index:05d}.npz" for index in range(4))
CHECKPOINT_NAME = "simple3d_synthetic.pt"


@dataclass(frozen=True)
class SceneSpec:
    name: str
    seed: int
    build: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grid() -> tuple[np.ndarray, np.ndarray]:
    axis_x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)
    axis_y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)
    return np.meshgrid(axis_x, axis_y, indexing="xy")


def _clip_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.stack((red, green, blue), axis=-1)
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _scene_ramp_sphere(xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    background = 4.5 + 0.75 * xx + 0.25 * yy
    radial_sq = ((xx + 0.28) / 0.47) ** 2 + ((yy - 0.05) / 0.47) ** 2
    sphere = 2.15 - 0.38 * np.sqrt(np.clip(1.0 - radial_sq, 0.0, 1.0))
    depth = np.where(radial_sq <= 1.0, sphere, background)
    red = 0.30 + 0.35 * (xx + 1.0) / 2.0
    green = 0.25 + 0.45 * (yy + 1.0) / 2.0
    blue = 0.30 + 0.55 * (radial_sq <= 1.0)
    return depth.astype(np.float32), _clip_rgb(red, green, blue)


def _scene_steps_checker(xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    column = np.floor((xx + 1.0) * 2.0).astype(np.int32)
    row = np.floor((yy + 1.0) * 2.0).astype(np.int32)
    column = np.clip(column, 0, 3)
    row = np.clip(row, 0, 3)
    depth = 1.65 + 0.72 * column + 0.38 * row
    checker = ((column + row) % 2).astype(np.float32)
    red = 0.18 + 0.62 * checker
    green = 0.68 - 0.42 * checker + 0.05 * column
    blue = 0.28 + 0.12 * row
    return depth.astype(np.float32), _clip_rgb(red, green, blue)


def _scene_waves_rings(xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.sqrt(xx**2 + yy**2)
    waves = 4.0 + 0.52 * np.sin(3.0 * math.pi * xx) * np.cos(2.0 * math.pi * yy)
    ring = (radius > 0.28) & (radius < 0.56)
    disk = radius <= 0.24
    depth = np.where(ring, 2.45 + 0.20 * np.sin(12.0 * radius), waves)
    depth = np.where(disk, 1.75 + 0.15 * radius, depth)
    red = 0.28 + 0.45 * (np.sin(2.0 * math.pi * radius) + 1.0) / 2.0
    green = 0.22 + 0.55 * (xx + 1.0) / 2.0
    blue = 0.75 - 0.40 * (yy + 1.0) / 2.0
    return depth.astype(np.float32), _clip_rgb(red, green, blue)


def _scene_pyramid_columns(xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pyramid = 2.25 + 2.15 * np.maximum(np.abs(xx), np.abs(yy))
    left_column = (xx + 0.52) ** 2 + (yy + 0.18) ** 2 < 0.13**2
    right_column = (xx - 0.48) ** 2 + (yy - 0.24) ** 2 < 0.18**2
    depth = np.where(left_column, 1.35, pyramid)
    depth = np.where(right_column, 1.85, depth)
    edge = np.maximum(np.abs(xx), np.abs(yy))
    red = 0.70 - 0.38 * edge + 0.18 * left_column
    green = 0.30 + 0.48 * (1.0 - edge) + 0.12 * right_column
    blue = 0.22 + 0.46 * edge
    return depth.astype(np.float32), _clip_rgb(red, green, blue)


SCENES = (
    SceneSpec("ramp_sphere", BASE_SEED + 0, _scene_ramp_sphere),
    SceneSpec("steps_checker", BASE_SEED + 1, _scene_steps_checker),
    SceneSpec("waves_rings", BASE_SEED + 2, _scene_waves_rings),
    SceneSpec("pyramid_columns", BASE_SEED + 3, _scene_pyramid_columns),
)


def _rgb_to_intensity_and_albedo(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_float = rgb.astype(np.float32) / 255.0
    intensity = (
        0.2126 * rgb_float[..., 0]
        + 0.7152 * rgb_float[..., 1]
        + 0.0722 * rgb_float[..., 2]
    )
    # Keep every material observable while preserving the procedural texture.
    albedo = 0.20 + 0.80 * (
        0.25 * rgb_float[..., 0]
        + 0.25 * rgb_float[..., 1]
        + 0.50 * rgb_float[..., 2]
    )
    return intensity.astype(np.float32), albedo.astype(np.float32)


def build_sample(index: int) -> dict[str, np.ndarray]:
    """Return one deterministic synthetic SPAD sample as safe NumPy arrays."""
    if not 0 <= index < len(SCENES):
        raise IndexError(f"sample index must be between 0 and {len(SCENES) - 1}")

    scene = SCENES[index]
    xx, yy = _grid()
    depth_m, rgb = scene.build(xx, yy)
    intensity, albedo = _rgb_to_intensity_and_albedo(rgb)

    if depth_m.shape != (HEIGHT, WIDTH) or rgb.shape != (HEIGHT, WIDTH, 3):
        raise RuntimeError(f"scene {scene.name} returned an invalid shape")
    if not np.isfinite(depth_m).all() or float(depth_m.min()) <= 0.0:
        raise RuntimeError(f"scene {scene.name} returned an invalid depth map")

    depth_bins = (2.0 * depth_m / LIGHT_SPEED_M_PER_S) / BIN_SIZE_SECONDS
    depth_bins = np.clip(depth_bins, 1.0, TEMPORAL_BINS - 2.0).astype(np.float32)

    signal = albedo / np.maximum(depth_m**2, np.float32(1e-6))
    signal *= np.float32(MEAN_SIGNAL_PHOTONS / float(signal.mean()))
    intensity_scale = intensity / max(float(intensity.mean()), 1e-6)
    background_per_bin = (
        intensity_scale * np.float32(MEAN_BACKGROUND_PHOTONS / TEMPORAL_BINS)
    ).astype(np.float32)

    time_axis = np.arange(TEMPORAL_BINS, dtype=np.float32)[:, None, None]
    pulse = np.exp(-0.5 * ((time_axis - depth_bins[None, ...]) / PULSE_SIGMA_BINS) ** 2)
    pulse /= np.maximum(pulse.sum(axis=0, keepdims=True), np.float32(1e-12))
    expectation = pulse * signal[None, ...] + background_per_bin[None, ...]

    rng = np.random.default_rng(scene.seed)
    sampled = rng.poisson(expectation).astype(np.uint16)
    peak_bin = np.argmax(sampled, axis=0).astype(np.float32)
    peak_signal = np.max(sampled, axis=0).astype(np.float32)
    xhat = np.stack((peak_bin, peak_signal, background_per_bin), axis=0).astype(np.float32)
    x_proxy = np.stack((depth_bins, signal, background_per_bin), axis=0).astype(np.float32)

    return {
        "counts": sampled,
        "depth_m": depth_m.astype(np.float32),
        "rgb": rgb,
        "albedo": albedo,
        "intensity": intensity,
        "xhat": xhat,
        "x": x_proxy,
        "surface_model": np.asarray("single"),
        "source_mode": np.asarray("synthetic"),
        "scene": np.asarray(scene.name),
        "sample_id": np.asarray(index, dtype=np.int32),
        "bin_size": np.asarray(BIN_SIZE_SECONDS, dtype=np.float64),
        "mean_signal_photons": np.asarray(MEAN_SIGNAL_PHOTONS, dtype=np.float32),
        "mean_background_photons": np.asarray(MEAN_BACKGROUND_PHOTONS, dtype=np.float32),
        "sbr": np.asarray(
            MEAN_SIGNAL_PHOTONS / MEAN_BACKGROUND_PHOTONS,
            dtype=np.float32,
        ),
        "generator_seed": np.asarray(scene.seed, dtype=np.int64),
        "asset_license": np.asarray("CC0-1.0"),
        "generator_version": np.asarray(GENERATOR_VERSION),
    }


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a byte-stable, pickle-free NPZ archive with fixed ZIP metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_stream:
        with zipfile.ZipFile(
            raw_stream,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(arrays):
                array = np.asanyarray(arrays[name])
                if array.dtype.hasobject:
                    raise TypeError(f"object arrays are forbidden in public demo field {name!r}")
                payload = io.BytesIO()
                np.lib.format.write_array(payload, array, allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _downsample_time(counts: np.ndarray, factor: int) -> np.ndarray:
    usable = (counts.shape[0] // factor) * factor
    return counts[:usable].reshape(usable // factor, factor, *counts.shape[1:]).sum(axis=1).astype(np.float32)


def _normalize_counts(counts: np.ndarray) -> np.ndarray:
    normalized = np.log1p(counts.astype(np.float32))
    return (normalized / max(float(normalized.max()), 1.0)).astype(np.float32)


def _target_distribution(depth_m: np.ndarray, temporal_downsample: int, sigma_bins: float) -> np.ndarray:
    bins = TEMPORAL_BINS // temporal_downsample
    centers = (2.0 * depth_m / LIGHT_SPEED_M_PER_S) / (BIN_SIZE_SECONDS * temporal_downsample)
    centers = np.clip(centers, 0.0, bins - 1.0).astype(np.float32)
    axis = np.arange(bins, dtype=np.float32)[:, None, None]
    target = np.exp(-0.5 * ((axis - centers[None, ...]) / sigma_bins) ** 2)
    target /= np.maximum(target.sum(axis=0, keepdims=True), np.float32(1e-12))
    return target.astype(np.float32)


def build_checkpoint(dataset_dir: Path, output_path: Path, sample_hashes: list[str]) -> dict[str, object]:
    """Train a tiny deterministic Simple3D model only on generated assets."""
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - exercised in minimal release builders
        raise RuntimeError(
            "PyTorch is required to reproduce the public Simple3D checkpoint; "
            "install the locked SPImaging training runtime first"
        ) from exc

    from spimaging.training_common.networks import build_model

    torch.manual_seed(BASE_SEED)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting this once per process.  Repeat generation
            # in an already-initialized process must remain usable.
            pass
    torch.use_deterministic_algorithms(True)

    temporal_downsample = 64
    spatial_stride = 2
    target_sigma_bins = 0.75
    epochs = 8
    learning_rate = 3e-3
    base_channels = 2

    model = build_model("simple3d", in_channels=1, base_channels=base_channels)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    losses: list[float] = []
    for _epoch in range(epochs):
        for sample_name in SAMPLE_NAMES:
            with np.load(dataset_dir / sample_name, allow_pickle=False) as archive:
                counts = _downsample_time(archive["counts"], temporal_downsample)
                depth_m = archive["depth_m"].astype(np.float32)
            counts = counts[:, ::spatial_stride, ::spatial_stride]
            depth_m = depth_m[::spatial_stride, ::spatial_stride]
            inputs = torch.from_numpy(_normalize_counts(counts)[None, None, ...])
            targets = torch.from_numpy(
                _target_distribution(depth_m, temporal_downsample, target_sigma_bins)[None, ...]
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs).squeeze(1)
            log_probs = functional.log_softmax(logits, dim=1)
            loss = functional.kl_div(log_probs, targets, reduction="none").sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

    model.eval()
    train_args = {
        "model": "simple3d",
        "base_channels": base_channels,
        "num_blocks": 1,
        "temporal_downsample": temporal_downsample,
        "target_sigma_bins": target_sigma_bins,
        "target_source": "depth",
        "no_log_counts": False,
        "seed": BASE_SEED,
        "epochs": epochs,
        "batch_size": 1,
        "lr": learning_rate,
        "weight_decay": 1e-5,
    }
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "epoch": epochs,
        "args": train_args,
        "model_name": "simple3d",
        "method_family": "supervised",
        "training_metrics": {
            "initial_kl": losses[0],
            "final_kl": losses[-1],
            "optimizer_steps": len(losses),
        },
        "training_sample_sha256": list(sample_hashes),
        "asset_license": "CC0-1.0",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "base_channels": base_channels,
        "temporal_downsample": temporal_downsample,
        "training_spatial_stride": spatial_stride,
        "target_sigma_bins": target_sigma_bins,
        "epochs": epochs,
        "optimizer": "AdamW",
        "optimizer_steps": len(losses),
        "learning_rate": learning_rate,
        "initial_kl": losses[0],
        "final_kl": losses[-1],
        "seed": BASE_SEED,
        "training_sample_sha256": list(sample_hashes),
    }


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "sample_id",
        "file",
        "scene",
        "source_mode",
        "surface_model",
        "height",
        "width",
        "temporal_bins",
        "seed",
        "mean_signal_photons",
        "mean_background_photons",
        "sbr",
        "sha256",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _asset_record(path: Path, relative_path: str) -> dict[str, object]:
    return {
        "path": relative_path.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _resolve_asset_path(output_root: Path, relative_path: str) -> Path:
    candidate = (output_root / relative_path).resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"manifest asset path escapes the public demo root: {relative_path!r}") from exc
    return candidate


def generate(output_root: Path, force: bool = False, samples_only: bool = False) -> Path:
    """Generate assets into *output_root* and return the manifest path."""
    output_root = output_root.resolve()
    dataset_dir = output_root / "dataset"
    checkpoint_dir = output_root / "checkpoint"
    manifest_path = output_root / "manifest.json"
    expected_targets = [dataset_dir / name for name in SAMPLE_NAMES]
    expected_targets.extend((dataset_dir / "index.csv", manifest_path))
    if not samples_only:
        expected_targets.append(checkpoint_dir / CHECKPOINT_NAME)

    existing = [path for path in expected_targets if path.exists()]
    if existing and not force:
        listed = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(f"refusing to replace existing public assets without --force: {listed}")

    unexpected_samples = sorted(
        path for path in dataset_dir.glob("sample_*.npz") if path.name not in SAMPLE_NAMES
    )
    if unexpected_samples:
        raise RuntimeError(
            "refusing to leave or remove unexpected sample files in the public dataset: "
            + ", ".join(path.name for path in unexpected_samples)
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spimaging-public-demo-", dir=output_root.parent) as temporary:
        staging_root = Path(temporary)
        staging_dataset = staging_root / "dataset"
        staging_checkpoint = staging_root / "checkpoint"
        rows: list[dict[str, object]] = []
        sample_records: list[dict[str, object]] = []
        sample_hashes: list[str] = []

        for index, (sample_name, scene) in enumerate(zip(SAMPLE_NAMES, SCENES)):
            sample_path = staging_dataset / sample_name
            write_deterministic_npz(sample_path, build_sample(index))
            digest = sha256_file(sample_path)
            sample_hashes.append(digest)
            record = _asset_record(sample_path, f"dataset/{sample_name}")
            record.update(
                {
                    "sample_id": index,
                    "scene": scene.name,
                    "seed": scene.seed,
                    "arrays": {
                        "counts": {"shape": [TEMPORAL_BINS, HEIGHT, WIDTH], "dtype": "uint16"},
                        "depth_m": {"shape": [HEIGHT, WIDTH], "dtype": "float32"},
                        "rgb": {"shape": [HEIGHT, WIDTH, 3], "dtype": "uint8"},
                    },
                }
            )
            sample_records.append(record)
            rows.append(
                {
                    "sample_id": index,
                    "file": sample_name,
                    "scene": scene.name,
                    "source_mode": "synthetic",
                    "surface_model": "single",
                    "height": HEIGHT,
                    "width": WIDTH,
                    "temporal_bins": TEMPORAL_BINS,
                    "seed": scene.seed,
                    "mean_signal_photons": MEAN_SIGNAL_PHOTONS,
                    "mean_background_photons": MEAN_BACKGROUND_PHOTONS,
                    "sbr": MEAN_SIGNAL_PHOTONS / MEAN_BACKGROUND_PHOTONS,
                    "sha256": digest,
                }
            )

        index_path = staging_dataset / "index.csv"
        _write_index(index_path, rows)

        checkpoint_record: dict[str, object]
        if samples_only:
            checkpoint_record = {
                "status": "not_generated",
                "reason": "--samples-only was selected",
                "recipe": "rerun without --samples-only in the locked training runtime",
            }
        else:
            checkpoint_path = staging_checkpoint / CHECKPOINT_NAME
            recipe = build_checkpoint(staging_dataset, checkpoint_path, sample_hashes)
            checkpoint_record = _asset_record(
                checkpoint_path,
                f"checkpoint/{CHECKPOINT_NAME}",
            )
            checkpoint_record.update(
                {
                    "status": "generated",
                    "model_name": "simple3d",
                    "method_family": "supervised",
                    "recipe": recipe,
                }
            )

        generator_path = Path(__file__).resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "release": "0.2.0-beta.1",
            "generator": {
                "name": "SPImaging analytic synthetic demo generator",
                "version": GENERATOR_VERSION,
                "path": "scripts/generate_synthetic_demo.py",
                "sha256": sha256_file(generator_path),
                "command": "python scripts/generate_synthetic_demo.py --force",
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "provenance": {
                "source_kind": "analytic_procedural_geometry",
                "external_source_files": [],
                "external_image_data": False,
                "description": (
                    "RGB, metric depth and SPAD counts are constructed from analytic "
                    "geometry and fixed pseudo-random seeds in the generator source."
                ),
                "asset_license": "CC0-1.0",
                "license_notice": "CC0_NOTICE.md",
            },
            "parameters": {
                "spatial_shape": [HEIGHT, WIDTH],
                "temporal_bins": TEMPORAL_BINS,
                "bin_size_seconds": BIN_SIZE_SECONDS,
                "mean_signal_photons": MEAN_SIGNAL_PHOTONS,
                "mean_background_photons": MEAN_BACKGROUND_PHOTONS,
                "pulse_sigma_bins": PULSE_SIGMA_BINS,
                "poisson_generator": "numpy.random.Generator(PCG64)",
                "base_seed": BASE_SEED,
            },
            "dataset": {
                "path": "dataset",
                "sample_count": len(sample_records),
                "index": _asset_record(index_path, "dataset/index.csv"),
                "samples": sample_records,
            },
            "checkpoint": checkpoint_record,
            "reproducibility": {
                "sample_archives": (
                    "Byte-stable through sorted NPY members, fixed ZIP metadata, explicit dtypes "
                    "and fixed PCG64 seeds."
                ),
                "checkpoint": (
                    "Deterministic CPU recipe; exact bytes are guaranteed for the locked release "
                    "Python/NumPy/PyTorch runtime recorded by the release build."
                ),
            },
        }
        staging_manifest = staging_root / "manifest.json"
        staging_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        output_root.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(staging_dataset.iterdir(), key=lambda path: path.name):
            os.replace(source, dataset_dir / source.name)
        if not samples_only:
            os.replace(staging_checkpoint / CHECKPOINT_NAME, checkpoint_dir / CHECKPOINT_NAME)
        os.replace(staging_manifest, manifest_path)

    return manifest_path


def verify(output_root: Path, require_checkpoint: bool = True) -> None:
    """Validate hashes, schemas, safe loading and checkpoint metadata."""
    output_root = output_root.resolve()
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    if manifest["provenance"].get("external_source_files") != []:
        raise ValueError("public demo manifest unexpectedly declares external source files")
    if manifest["provenance"].get("external_image_data") is not False:
        raise ValueError("public demo manifest must explicitly reject external image data")
    if manifest["generator"].get("sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("generator source hash does not match the public manifest")

    index_record = manifest["dataset"]["index"]
    index_path = _resolve_asset_path(output_root, index_record["path"])
    if index_path.stat().st_size != index_record["bytes"] or sha256_file(index_path) != index_record["sha256"]:
        raise ValueError("dataset index hash or size mismatch")

    samples = manifest["dataset"]["samples"]
    if len(samples) != 4:
        raise ValueError(f"expected four samples, found {len(samples)}")
    for record in samples:
        path = _resolve_asset_path(output_root, record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"hash or size mismatch: {record['path']}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "counts",
                "depth_m",
                "rgb",
                "surface_model",
                "source_mode",
                "generator_seed",
                "asset_license",
            }
            if not required.issubset(archive.files):
                raise ValueError(f"required fields are missing: {record['path']}")
            if archive["counts"].shape != (TEMPORAL_BINS, HEIGHT, WIDTH):
                raise ValueError(f"invalid counts shape: {record['path']}")
            if archive["counts"].dtype != np.uint16:
                raise ValueError(f"invalid counts dtype: {record['path']}")
            if archive["depth_m"].shape != (HEIGHT, WIDTH):
                raise ValueError(f"invalid depth shape: {record['path']}")
            if archive["depth_m"].dtype != np.float32:
                raise ValueError(f"invalid depth dtype: {record['path']}")
            if archive["rgb"].shape != (HEIGHT, WIDTH, 3):
                raise ValueError(f"invalid RGB shape: {record['path']}")
            if archive["rgb"].dtype != np.uint8:
                raise ValueError(f"invalid RGB dtype: {record['path']}")
            for field in archive.files:
                if archive[field].dtype.hasobject:
                    raise ValueError(f"object array is forbidden: {record['path']}:{field}")
                if archive[field].dtype.kind in "fiu" and not np.isfinite(archive[field]).all():
                    raise ValueError(f"non-finite numeric array: {record['path']}:{field}")
            if str(archive["source_mode"]) != "synthetic":
                raise ValueError(f"invalid source mode: {record['path']}")
            if str(archive["asset_license"]) != "CC0-1.0":
                raise ValueError(f"invalid asset license: {record['path']}")

    checkpoint = manifest["checkpoint"]
    if require_checkpoint:
        if checkpoint.get("status") != "generated":
            raise ValueError("public checkpoint was not generated")
        checkpoint_path = _resolve_asset_path(output_root, checkpoint["path"])
        if (
            checkpoint_path.stat().st_size != checkpoint["bytes"]
            or sha256_file(checkpoint_path) != checkpoint["sha256"]
        ):
            raise ValueError("checkpoint hash or size mismatch")
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("model_name") != "simple3d" or payload.get("method_family") != "supervised":
            raise ValueError("checkpoint metadata is invalid")
        if payload.get("training_sample_sha256") != [item["sha256"] for item in samples]:
            raise ValueError("checkpoint training-input hashes do not match the public samples")


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate deterministic, CC0 SPImaging public demo assets without external data."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root / "public_demo",
        help="Destination root (default: repository public_demo directory).",
    )
    parser.add_argument("--force", action="store_true", help="Replace the known generated files.")
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Reproduce samples/index without importing PyTorch or producing a checkpoint.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the existing manifest and assets without writing files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        verify(args.output_root, require_checkpoint=not args.samples_only)
        print(f"Verified public demo assets: {args.output_root.resolve()}")
        return 0

    manifest_path = generate(args.output_root, force=args.force, samples_only=args.samples_only)
    verify(args.output_root, require_checkpoint=not args.samples_only)
    print(f"Generated and verified public demo manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
