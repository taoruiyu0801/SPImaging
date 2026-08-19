"""Day 22 end-to-end checks for all four SPAD measurement models."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

import imageio.v3 as iio
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMON_KEYS = {
    "counts",
    "depth_m",
    "rgb",
    "albedo",
    "intensity",
    "xhat",
    "mean_signal_photons",
    "mean_background_photons",
    "sbr",
    "bins",
    "bin_size",
    "surface_model",
    "source_mode",
    "scene",
    "sample_id",
}
MODEL_KEYS = {
    "single": set(),
    "neighborhood_mix": {"x", "transient_clean"},
    "translucent_layer": {
        "x",
        "transient_clean",
        "front_depth_m",
        "front_signal",
        "back_signal_after_transmission",
        "front_tof_bin",
        "back_tof_bin",
    },
    "volume_scattering": {
        "x",
        "transient_clean",
        "volume_depth_limit_m",
        "volume_scatter_signal",
        "surface_signal_after_medium",
        "volume_scatter_tof_map",
        "surface_tof_bin",
        "volume_medium_type_id",
    },
}


def create_middlebury_fixture(root: Path) -> Path:
    scene = root / "scene_01"
    scene.mkdir(parents=True)
    height = width = 12
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (xx * 17 + 20) % 256,
            (yy * 19 + 30) % 256,
            ((xx + yy) * 11 + 40) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    disparity = (32 + xx + yy).astype(np.uint16)
    iio.imwrite(scene / "view1.png", rgb)
    iio.imwrite(scene / "disp1.png", disparity)
    return root


def run_generation(model: str, source_root: Path, output_dir: Path, cwd: Path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH"))
        if part
    )
    arguments = [
        sys.executable,
        "-m",
        "spimaging.generation.pipeline",
        "--dataset_mode",
        "middlebury",
        "--middlebury_root",
        str(source_root),
        "--output_dir",
        str(output_dir),
        "--surface_model",
        model,
        "--param_idx",
        "3",
        "--res",
        "8",
        "--bins",
        "64",
        "--bin_size",
        "1e-9",
        "--limit",
        "1",
        "--mix_kernel_size",
        "3",
        "--volume_num_steps",
        "8",
        "--save_x",
        "--save_clean_transient",
    ]
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )


@pytest.mark.integration
@pytest.mark.parametrize("model", tuple(MODEL_KEYS))
def test_all_generation_models_publish_expected_artifacts(model: str, tmp_path: Path) -> None:
    source_root = create_middlebury_fixture(tmp_path / "middlebury")
    output_dir = tmp_path / f"generated_{model}"

    result = run_generation(model, source_root, output_dir, tmp_path)
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert "Traceback (most recent call last)" not in combined
    assert output_dir.is_dir()
    samples = sorted(output_dir.glob("sample_*.npz"))
    assert len(samples) == 1
    assert (output_dir / "index.csv").is_file()

    with (output_dir / "index.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["surface_model"] == model

    with np.load(samples[0], allow_pickle=False) as sample:
        keys = set(sample.files)
        assert COMMON_KEYS.issubset(keys)
        assert MODEL_KEYS[model].issubset(keys)
        assert sample["counts"].shape == (64, 8, 8)
        assert sample["depth_m"].shape == (8, 8)
        assert sample["xhat"].shape == (3, 8, 8)
        assert str(sample["surface_model"]) == model
        assert np.isfinite(sample["counts"]).all()
        assert np.isfinite(sample["depth_m"]).all()
