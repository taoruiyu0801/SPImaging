"""Day 25 integration checks for checkpoints, prediction, and evaluation."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATASET = REPOSITORY_ROOT / "example_data" / "nyuv2_raw_single_random_snr"
DEMO_SAMPLE = DEMO_DATASET / "sample_00000.npz"
DEMO_CHECKPOINT = REPOSITORY_ROOT / "demo_checkpoint" / "simple3d_demo_best.pt"


def run_module(module_name: str, arguments: list[str], cwd: Path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH"))
        if part
    )
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", module_name, *arguments],
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
def test_correct_checkpoint_single_prediction(tmp_path: Path) -> None:
    output_npz = tmp_path / "predict" / "prediction.npz"
    output_figure = tmp_path / "predict" / "comparison.png"
    result = run_module(
        "spimaging.testing.predict",
        [
            "--checkpoint",
            str(DEMO_CHECKPOINT),
            "--sample_file",
            str(DEMO_SAMPLE),
            "--output_npz",
            str(output_npz),
            "--output_fig",
            str(output_figure),
        ],
        tmp_path,
    )
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert "Traceback (most recent call last)" not in combined
    assert output_npz.is_file()
    assert output_figure.is_file()
    with np.load(output_npz, allow_pickle=False) as prediction:
        assert {
            "pred_depth_m",
            "target_depth_m",
            "abs_error_m",
            "method_family",
        }.issubset(prediction.files)
        assert prediction["pred_depth_m"].shape == (64, 64)
        assert prediction["target_depth_m"].shape == (64, 64)
        assert prediction["abs_error_m"].shape == (64, 64)
        assert str(prediction["method_family"]) == "supervised"


@pytest.mark.integration
def test_correct_checkpoint_batch_evaluation(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluate"
    result = run_module(
        "spimaging.testing.evaluate",
        [
            "--checkpoint",
            str(DEMO_CHECKPOINT),
            "--label",
            "day25-simple3d",
            "--dataset_dir",
            str(DEMO_DATASET),
            "--output_dir",
            str(output_dir),
        ],
        tmp_path,
    )
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert "Traceback (most recent call last)" not in combined
    with (output_dir / "metrics_per_sample.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 4
    assert {row["sample"] for row in rows} == {f"sample_{index:05d}.npz" for index in range(4)}
    assert summary["day25-simple3d"]["n_samples"] == 4
    assert (output_dir / "comparison.png").is_file()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("module_name", "arguments", "output_relative"),
    (
        (
            "spimaging.testing.predict",
            [
                "--checkpoint",
                "{checkpoint}",
                "--sample_file",
                str(DEMO_SAMPLE),
                "--output_npz",
                "predict/prediction.npz",
            ],
            "predict",
        ),
        (
            "spimaging.testing.evaluate",
            [
                "--checkpoint",
                "{checkpoint}",
                "--dataset_dir",
                str(DEMO_DATASET),
                "--output_dir",
                "evaluate",
            ],
            "evaluate",
        ),
    ),
)
def test_corrupt_checkpoint_is_rejected_without_outputs(
    module_name: str,
    arguments: list[str],
    output_relative: str,
    tmp_path: Path,
) -> None:
    corrupt_checkpoint = tmp_path / "corrupt.pt"
    corrupt_checkpoint.write_bytes(b"not a torch checkpoint")
    resolved_arguments = [
        str(corrupt_checkpoint) if argument == "{checkpoint}" else argument
        for argument in arguments
    ]

    result = run_module(module_name, resolved_arguments, tmp_path)
    combined = result.stdout + result.stderr

    assert result.returncode == 2, combined
    assert "cannot load --checkpoint" in combined
    assert "Traceback (most recent call last)" not in combined
    assert not (tmp_path / output_relative).exists()
