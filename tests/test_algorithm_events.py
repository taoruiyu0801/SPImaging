"""T05 structured progress and cooperative training tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spimaging.supervised_training import train as supervised
from spimaging.self_supervised_training import train as self_supervised
from spimaging.training_common.events import (
    EVENT_PREFIX,
    CancellationRequested,
    cancellation_requested,
    emit_event,
    raise_if_cancelled,
)
from spimaging.training_common.security import load_checkpoint_safely


def _dataset(directory: Path) -> Path:
    directory.mkdir()
    for index in range(2):
        np.savez_compressed(
            directory / f"sample_{index:05d}.npz",
            counts=np.ones((4, 4, 4), dtype=np.float32),
            depth_m=np.full((4, 4), 0.02, dtype=np.float32),
            bin_size=np.asarray(80e-12, dtype=np.float32),
        )
    return directory


def test_structured_event_prefix_is_opt_in_and_callback_always_receives_event(
    monkeypatch, capsys
) -> None:
    captured = []
    emit_event("batch", callback=captured.append, batch=1)
    assert capsys.readouterr().out == ""
    assert captured == [{"type": "batch", "batch": 1}]

    monkeypatch.setenv("SPIMAGING_STRUCTURED_EVENTS", "1")
    emit_event("epoch", epoch=2, loss=0.5)
    line = capsys.readouterr().out.strip()
    assert line.startswith(EVENT_PREFIX)
    assert json.loads(line[len(EVENT_PREFIX) :]) == {"type": "epoch", "epoch": 2, "loss": 0.5}


def test_cancel_file_and_callback_are_cooperative(tmp_path: Path, monkeypatch) -> None:
    cancel_file = tmp_path / "cancel.request"
    monkeypatch.setenv("SPIMAGING_CANCEL_FILE", str(cancel_file))
    assert not cancellation_requested()
    cancel_file.touch()
    assert cancellation_requested()
    with pytest.raises(CancellationRequested):
        raise_if_cancelled(epoch=1, next_batch=2)
    assert cancellation_requested(lambda: True)


def test_worker_device_environment_becomes_cli_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SPIMAGING_DEVICE", "cpu")
    monkeypatch.setenv("SPIMAGING_GPU_INDEX", "3")
    args = supervised.build_parser().parse_args(["--dataset_dir", "placeholder"])
    assert args.device == "cpu"
    assert args.gpu_index == 3


def test_supervised_cancel_checkpoint_can_resume_and_writes_history(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "training"
    common = [
        "--dataset_dir",
        str(dataset),
        "--output_dir",
        str(output),
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--base_channels",
        "1",
        "--num_blocks",
        "1",
        "--val_fraction",
        "0",
        "--device",
        "cpu",
    ]

    events = []
    with pytest.raises(SystemExit) as cancelled:
        supervised.main(common, event_callback=events.append, cancel_check=lambda: True)
    assert cancelled.value.code == 130
    assert events[-1]["type"] == "cancelled"
    checkpoint_path = output / "cancelled.pt"
    checkpoint = load_checkpoint_safely(checkpoint_path)
    assert checkpoint["status"] == "cancelled"
    assert checkpoint["resume"]["next_epoch"] == 1
    assert checkpoint["resume"]["next_batch"] == 0

    supervised.main(
        [*common, "--resume_checkpoint", str(checkpoint_path)],
        event_callback=events.append,
        cancel_check=lambda: False,
    )
    assert (output / "last.pt").is_file()
    assert (output / "best.pt").is_file()
    assert len((output / "training_history.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert events[-1]["type"] == "completed"


def test_self_supervised_cancel_emits_safe_resumable_checkpoint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "self-supervised"
    events = []
    with pytest.raises(SystemExit) as cancelled:
        self_supervised.main(
            [
                "--dataset_dir",
                str(dataset),
                "--output_dir",
                str(output),
                "--epochs",
                "1",
                "--base_channels",
                "1",
                "--num_blocks",
                "1",
                "--time_scale",
                "1",
                "--spatial_scale",
                "1",
                "--temporal_downsample",
                "1",
                "--spatial_downsample",
                "1",
                "--device",
                "cpu",
            ],
            event_callback=events.append,
            cancel_check=lambda: True,
        )
    assert cancelled.value.code == 130
    checkpoint = load_checkpoint_safely(output / "cancelled.pt")
    assert checkpoint["method_family"] == "self_supervised_spisr"
    assert checkpoint["resume"]["next_epoch"] == 1
    assert events[-1]["type"] == "cancelled"
