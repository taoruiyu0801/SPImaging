"""CUDA-only external engine selection and self-test contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from launcher.engine import (
    EngineProbe,
    NvidiaDevice,
    engine_environment,
    probe_engine,
    select_cuda_profile,
)
from launcher.errors import LauncherError


def test_blackwell_selects_cu128_profile() -> None:
    profile = select_cuda_profile(NvidiaDevice("RTX 5070 Ti", "610.62", (12, 0)))
    assert profile.profile_id == "torch-2.9.1-cu128"
    assert profile.index_url.endswith("/cu128")


def test_older_gpu_and_driver_select_cu126_profile() -> None:
    profile = select_cuda_profile(NvidiaDevice("RTX 3080", "565.90", (8, 6)))
    assert profile.profile_id == "torch-2.7.1-cu126"


def test_blackwell_old_driver_is_rejected_instead_of_cpu_fallback() -> None:
    with pytest.raises(LauncherError, match="至少需要驱动 570.65"):
        select_cuda_profile(NvidiaDevice("RTX 5070 Ti", "566.10", (12, 0)))


def test_probe_requires_real_cuda_workload_and_parses_result(tmp_path: Path) -> None:
    python = tmp_path / "python.exe"
    python.touch()
    app = tmp_path / "app"
    (app / "spimaging").mkdir(parents=True)
    (app / "spimaging" / "__init__.py").touch()
    payload = {
        "python_version": "3.12.10",
        "torch_version": "2.9.1+cu128",
        "cuda_version": "12.8",
        "device_name": "RTX 5070 Ti",
        "compute_capability": [12, 0],
        "free_memory": 10,
        "total_memory": 12,
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")
    with patch("launcher.engine.subprocess.run", return_value=completed) as mocked:
        result = probe_engine(python, app)
    assert result.compatible
    assert result.compute_capability == (12, 0)
    probe_program = mocked.call_args.args[0][2]
    assert "conv3d" in probe_program
    assert "torch.cuda.synchronize" in probe_program


def test_failed_cuda_kernel_probe_is_not_accepted(tmp_path: Path) -> None:
    python = tmp_path / "python.exe"
    python.touch()
    app = tmp_path / "app"
    app.mkdir()
    completed = subprocess.CompletedProcess(
        [], 1, "", "RuntimeError: no kernel image is available for execution on the device\n"
    )
    with patch("launcher.engine.subprocess.run", return_value=completed):
        result = probe_engine(python, app)
    assert not result.compatible
    assert "no kernel image" in result.reason


def test_engine_environment_isolates_user_python_and_marks_cuda_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "engine"
    scripts = prefix / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    python.touch()
    app = tmp_path / "app"
    app.mkdir()
    monkeypatch.setenv("PYTHONHOME", "C:/unsafe")
    monkeypatch.setenv("PYTHONPATH", "C:/shadow")
    monkeypatch.setenv("VIRTUAL_ENV", "C:/other")
    probe = EngineProbe(str(python), True, "ok")
    environment = engine_environment(probe, app)
    assert environment["PYTHONPATH"] == str(app.resolve())
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["SPIMAGING_CUDA_REQUIRED"] == "1"
    assert environment["SPIMAGING_RUNTIME_VARIANT"] == "cuda"
    assert "PYTHONHOME" not in environment
    assert "VIRTUAL_ENV" not in environment
