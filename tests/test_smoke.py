"""Day 21 pytest smoke checks for the public SPImaging package."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATASET = REPOSITORY_ROOT / "example_data" / "nyuv2_raw_single_random_snr"
PUBLIC_MODULES = (
    "spimaging.generation.pipeline",
    "spimaging.supervised_training.train",
    "spimaging.self_supervised_training.train",
    "spimaging.testing.predict",
    "spimaging.testing.evaluate",
    "spimaging.testing.verify",
    "spimaging.testing.browse",
    "spimaging.demo",
)


def test_package_import() -> None:
    import spimaging

    assert spimaging.__version__ == "0.2.0-beta.1"
    assert Path(spimaging.__file__).resolve().is_relative_to(REPOSITORY_ROOT)


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_command_help(module_name: str, tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH"))
        if part
    )
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / ".matplotlib")

    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "usage:" in result.stdout.lower(), combined
    assert "Traceback (most recent call last)" not in combined


def test_demo_sample_is_readable() -> None:
    sample_path = DEMO_DATASET / "sample_00000.npz"
    with np.load(sample_path, allow_pickle=False) as sample:
        assert {"counts", "depth_m", "surface_model"}.issubset(sample.files)
        assert sample["counts"].shape == (1024, 64, 64)
        assert sample["depth_m"].shape == (64, 64)
        assert np.isfinite(sample["counts"]).all()
        assert np.isfinite(sample["depth_m"]).all()
