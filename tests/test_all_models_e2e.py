"""CPU-minimal train -> predict -> evaluate regression for every public model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATASET = REPOSITORY_ROOT / "public_demo" / "dataset"


def _run(module: str, arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(cwd / ".matplotlib-cache")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH"))
        if part
    )
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
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
    ("model", "training_module", "method_family", "model_arguments"),
    (
        ("simple3d", "spimaging.supervised_training.train", "supervised", ["--base_channels", "1"]),
        ("prsnet", "spimaging.supervised_training.train", "supervised", ["--num_blocks", "1"]),
        ("penonlocal", "spimaging.supervised_training.train", "supervised", ["--num_blocks", "1"]),
        ("stin", "spimaging.supervised_training.train", "supervised", []),
        (
            "spisr",
            "spimaging.self_supervised_training.train",
            "self_supervised_spisr",
            [
                "--base_channels", "1",
                "--num_blocks", "1",
                "--time_scale", "1",
                "--spatial_scale", "1",
                "--spatial_downsample", "1",
                "--max_shift", "0",
            ],
        ),
    ),
)
def test_every_reconstruction_model_completes_train_predict_evaluate(
    model: str,
    training_module: str,
    method_family: str,
    model_arguments: list[str],
    tmp_path: Path,
) -> None:
    training_dataset = tmp_path / "training-data"
    evaluation_dataset = tmp_path / "evaluation-data"
    training_dataset.mkdir()
    evaluation_dataset.mkdir()
    for index in range(2):
        shutil.copy2(
            PUBLIC_DATASET / f"sample_{index:05d}.npz",
            training_dataset / f"sample_{index:05d}.npz",
        )
    shutil.copy2(
        PUBLIC_DATASET / "sample_00000.npz",
        evaluation_dataset / "sample_00000.npz",
    )

    train_dir = tmp_path / "train"
    train = _run(
        training_module,
        [
            "--dataset_dir", str(training_dataset),
            "--output_dir", str(train_dir),
            "--epochs", "1",
            "--batch_size", "1",
            "--max_samples", "2",
            "--val_fraction", "0.5",
            "--model", model,
            "--temporal_downsample", "64",
            "--num_workers", "0",
            "--seed", "0",
            "--device", "cpu",
            *model_arguments,
        ],
        tmp_path,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    checkpoint = train_dir / "best.pt"
    assert checkpoint.is_file()

    import torch

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint_data["model_name"] == model
    assert checkpoint_data["method_family"] == method_family

    prediction_path = tmp_path / "predict" / "prediction.npz"
    prediction = _run(
        "spimaging.testing.predict",
        [
            "--checkpoint", str(checkpoint),
            "--sample_file", str(evaluation_dataset / "sample_00000.npz"),
            "--output_npz", str(prediction_path),
            "--device", "cpu",
        ],
        tmp_path,
    )
    assert prediction.returncode == 0, prediction.stdout + prediction.stderr
    with np.load(prediction_path, allow_pickle=False) as archive:
        assert archive["pred_depth_m"].shape == (64, 64)
        assert np.isfinite(archive["pred_depth_m"]).all()
        assert str(archive["method_family"]) == method_family

    metrics_dir = tmp_path / "evaluate"
    evaluation = _run(
        "spimaging.testing.evaluate",
        [
            "--checkpoint", str(checkpoint),
            "--label", model,
            "--dataset_dir", str(evaluation_dataset),
            "--output_dir", str(metrics_dir),
            "--device", "cpu",
        ],
        tmp_path,
    )
    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    summary = json.loads((metrics_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    progress = json.loads((metrics_dir / "evaluation_progress.json").read_text(encoding="utf-8"))
    assert summary[model]["n_samples"] == 1
    assert summary[model]["complete"] is True
    assert progress["status"] == "complete"
