"""Real subprocess coverage for worker-to-algorithm structured events."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import imageio.v3 as iio
import numpy as np
import pytest

from spimaging.appcore.config import RunConfig
from spimaging.training_common.events import EVENT_PREFIX


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_worker(config: RunConfig) -> list[dict]:
    run_dir = Path(config.output.run_dir)
    run_dir.mkdir(parents=True)
    config_path = run_dir / "run.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH"))
        if item
    )
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, "-u", "-m", "spimaging.worker", "--config", str(config_path)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert events
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert {event["run_id"] for event in events} == {config.run_id}
    assert not any(
        EVENT_PREFIX in str(event.get("payload", {}).get("message", ""))
        for event in events
    ), "child structured events must not be downgraded to worker log events"
    return events


def _training_dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    for index in range(2):
        counts = np.ones((8, 4, 4), dtype=np.float32)
        counts[(index + 2) % 8] += 2.0
        np.savez_compressed(
            root / f"sample_{index:05d}.npz",
            counts=counts,
            depth_m=np.full((4, 4), 0.02 + index * 0.001, dtype=np.float32),
            bin_size=np.asarray(80e-12, dtype=np.float32),
        )
    return root


def _middlebury_fixture(root: Path) -> Path:
    yy, xx = np.mgrid[:12, :12]
    for index in range(2):
        scene = root / f"scene_{index:02d}"
        scene.mkdir(parents=True)
        rgb = np.stack(
            (
                (xx * 17 + 20 + index) % 256,
                (yy * 19 + 30 + index) % 256,
                ((xx + yy) * 11 + 40 + index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        disparity = (32 + xx + yy + index).astype(np.uint16)
        iio.imwrite(scene / "view1.png", rgb)
        iio.imwrite(scene / "disp1.png", disparity)
    return root


@pytest.mark.integration
def test_real_worker_preserves_supervised_batch_events(tmp_path: Path) -> None:
    dataset = _training_dataset(tmp_path / "training data with spaces")
    config = RunConfig.new(
        "train",
        tmp_path / "training worker run with spaces",
        input={"dataset_paths": [str(dataset)]},
        training={
            "enabled": True,
            "model": "simple3d",
            "preset": "quick",
            "parameters": {
                "epochs": 1,
                "batch_size": 1,
                "val_fraction": 0.0,
                "max_samples": 2,
                "base_channels": 1,
                "temporal_downsample": 1,
            },
        },
        compute={"preference": "cpu"},
        output={"history_db": str(tmp_path / "history.sqlite3")},
    )

    events = _run_worker(config)

    batches = [
        event
        for event in events
        if event["type"] == "batch" and event["payload"].get("phase") == "train"
    ]
    assert [event["payload"]["global_step"] for event in batches] == [1]
    validation_batches = [
        event
        for event in events
        if event["type"] == "batch"
        and event["payload"].get("phase") == "validation"
    ]
    assert len(validation_batches) == 1
    assert events[-1]["type"] == "completed"
    assert events[-1]["payload"]["status"] == "succeeded"


@pytest.mark.integration
def test_real_worker_preserves_generation_sample_events(tmp_path: Path) -> None:
    fixture = _middlebury_fixture(tmp_path / "middlebury fixture with spaces")
    config = RunConfig.new(
        "generate",
        tmp_path / "generation worker run with spaces",
        input={"source_path": str(fixture)},
        generation={
            "enabled": True,
            "dataset_mode": "middlebury",
            "surface_model": "neighborhood_mix",
            "parameters": {
                "param_idx": 3,
                "res": 8,
                "bins": 32,
                "bin_size": 1e-9,
                "limit": 2,
                "mix_kernel_size": 3,
            },
        },
        compute={"preference": "cpu"},
        output={"history_db": str(tmp_path / "history.sqlite3")},
    )

    events = _run_worker(config)

    samples = [
        event
        for event in events
        if event["type"] == "sample" and event["payload"].get("phase") == "generation"
    ]
    assert [event["payload"]["sample_id"] for event in samples] == [0, 1]
    assert events[-1]["type"] == "completed"
    assert events[-1]["payload"]["status"] == "succeeded"
