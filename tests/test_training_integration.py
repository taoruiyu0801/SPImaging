"""Day 24 integration checks for training, CPU fallback, and checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATASET = REPOSITORY_ROOT / "example_data" / "nyuv2_raw_single_random_snr"


def run_training(
    output_dir: Path,
    cwd: Path,
    *,
    batch_size: int,
    max_samples: int,
    force_cpu: bool,
):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH"))
        if part
    )
    if force_cpu:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "spimaging.supervised_training.train",
            "--dataset_dir",
            str(DEMO_DATASET),
            "--output_dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch_size",
            str(batch_size),
            "--max_samples",
            str(max_samples),
            "--val_fraction",
            "0.5" if max_samples > 1 else "0",
            "--model",
            "simple3d",
            "--base_channels",
            "2",
            "--temporal_downsample",
            "64",
            "--tv_weight",
            "0.005",
            "--num_workers",
            "0",
            "--seed",
            "0",
        ],
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
@pytest.mark.parametrize(
    ("case_name", "batch_size", "max_samples", "force_cpu"),
    (
        ("minimal-sample", 1, 1, False),
        ("batch-size-2", 2, 4, False),
        ("cpu-fallback", 1, 2, True),
    ),
)
def test_simple3d_training_matrix(
    case_name: str,
    batch_size: int,
    max_samples: int,
    force_cpu: bool,
    tmp_path: Path,
) -> None:
    import torch

    output_dir = tmp_path / case_name
    result = run_training(
        output_dir,
        tmp_path,
        batch_size=batch_size,
        max_samples=max_samples,
        force_cpu=force_cpu,
    )
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert "Traceback (most recent call last)" not in combined
    assert "Model: simple3d" in result.stdout
    if force_cpu:
        assert "Device: cpu" in result.stdout
    else:
        assert "Device: " in result.stdout

    for checkpoint_name in ("last.pt", "best.pt"):
        checkpoint_path = output_dir / checkpoint_name
        assert checkpoint_path.is_file()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 1
        assert checkpoint["model_name"] == "simple3d"
        assert checkpoint["method_family"] == "supervised"
        assert checkpoint["args"]["batch_size"] == batch_size
        assert checkpoint["args"]["max_samples"] == max_samples
        assert checkpoint["model_state"]
        assert checkpoint["optimizer_state"]


@pytest.mark.integration
@pytest.mark.parametrize("model_name", ("prsnet", "penonlocal", "stin"))
def test_other_supervised_models_can_start(model_name: str) -> None:
    import torch

    from spimaging.training_common.networks import build_model

    model = build_model(model_name, in_channels=1, base_channels=1, num_blocks=1)
    sample = torch.zeros((1, 1, 16, 4, 4), dtype=torch.float32)
    with torch.no_grad():
        output = model(sample)

    assert output.shape == sample.shape
    assert torch.isfinite(output).all()
