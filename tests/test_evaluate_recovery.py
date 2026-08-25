"""Durability checks for incremental evaluation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spimaging.testing import evaluate


def _sample(path: Path, value: float) -> None:
    np.savez_compressed(
        path,
        counts=np.ones((4, 4, 8), dtype=np.uint16),
        depth_m=np.full((4, 4), value, dtype=np.float32),
    )


def test_failure_retains_completed_rows_and_marks_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _sample(dataset / "sample_00000.npz", 1.0)
    _sample(dataset / "sample_00001.npz", 2.0)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")
    output = tmp_path / "metrics"

    monkeypatch.setattr(
        "spimaging.training_common.device.get_torch_device",
        lambda **_kwargs: SimpleNamespace(device="cpu", fallback=False, reason="CPU"),
    )
    monkeypatch.setattr(
        "spimaging.testing.predict.load_model",
        lambda *_args, **_kwargs: (object(), {}, "supervised"),
    )
    calls = 0

    def fake_predict_one(_checkpoint, sample, _device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic evaluation failure")
        with np.load(sample, allow_pickle=False) as archive:
            target = np.asarray(archive["depth_m"], dtype=np.float32)
        return target + 0.25, target

    monkeypatch.setattr(evaluate, "predict_one", fake_predict_one)
    monkeypatch.setattr(
        "sys.argv",
        [
            "spad-evaluate",
            "--checkpoint",
            str(checkpoint),
            "--label",
            "partial-model",
            "--dataset_dir",
            str(dataset),
            "--output_dir",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        evaluate.main()

    assert raised.value.code == 2
    with (output / "metrics_per_sample.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["sample"] for row in rows] == ["sample_00000.npz"]
    summary = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["partial-model"]["n_samples"] == 1
    assert summary["partial-model"]["complete"] is False
    progress = json.loads((output / "evaluation_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["completed_rows"] == 1
    assert "synthetic evaluation failure" in progress["error"]
    assert not (output / "comparison.png").exists()
