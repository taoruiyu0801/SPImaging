"""CUDA-only external engine selection and self-test contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from launcher.app import (
    _gui_requested_engine_choice,
    _load_engine_choice,
    _run_engine_headless,
    _save_engine_choice,
    build_parser,
)
from launcher.engine import (
    CudaEngineManager,
    CPU_PROFILE,
    EngineProbe,
    EngineProgress,
    NvidiaDevice,
    _install_environment,
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


def test_cpu_profile_and_real_conv3d_probe_are_separate_from_cuda(tmp_path: Path) -> None:
    assert CPU_PROFILE.index_url.endswith("/cpu")
    python = tmp_path / "python.exe"
    python.touch()
    app = tmp_path / "app"
    app.mkdir()
    payload = {
        "variant": "cpu",
        "python_version": "3.12.10",
        "torch_version": "2.9.1+cpu",
        "cuda_version": "",
        "device_name": "CPU",
        "compute_capability": [0, 0],
        "free_memory": 0,
        "total_memory": 0,
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")
    with patch("launcher.engine.subprocess.run", return_value=completed) as mocked:
        result = probe_engine(python, app, variant="cpu")
    assert result.compatible
    assert result.variant == "cpu"
    assert result.cuda_version == ""
    assert result.reason == "CPU 张量与 3D 卷积自检通过"
    command = mocked.call_args.args[0]
    assert command[-1] == "cpu"
    assert "conv3d" in command[2]


def test_cpu_engine_environment_marks_cpu_runtime(tmp_path: Path) -> None:
    prefix = tmp_path / "engine"
    scripts = prefix / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    python.touch()
    app = tmp_path / "app"
    app.mkdir()
    environment = engine_environment(EngineProbe(str(python), True, "ok", variant="cpu"), app)
    assert environment["SPIMAGING_CUDA_REQUIRED"] == "0"
    assert environment["SPIMAGING_RUNTIME_VARIANT"] == "cpu"


def test_first_launch_choice_is_persisted_without_touching_user_conda(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--install-root", str(tmp_path)])
    assert _load_engine_choice(args) is None
    _save_engine_choice(tmp_path, "cpu")
    assert _load_engine_choice(args) == "cpu"
    payload = json.loads((tmp_path / "metadata" / "engine-choice.json").read_text(encoding="utf-8"))
    assert payload["variant"] == "cpu"


def test_gui_ignores_stale_choice_unless_runtime_is_explicit(tmp_path: Path) -> None:
    _save_engine_choice(tmp_path, "cuda")
    ordinary = build_parser().parse_args(["--install-root", str(tmp_path)])
    automatic = build_parser().parse_args(
        ["--install-root", str(tmp_path), "--runtime", "auto"]
    )
    explicit_cpu = build_parser().parse_args(
        ["--install-root", str(tmp_path), "--runtime", "cpu"]
    )
    explicit_cuda = build_parser().parse_args(
        ["--install-root", str(tmp_path), "--runtime", "cuda"]
    )

    assert _load_engine_choice(ordinary) == "cuda"
    assert _gui_requested_engine_choice(ordinary) is None
    assert _gui_requested_engine_choice(automatic) is None
    assert _gui_requested_engine_choice(explicit_cpu) == "cpu"
    assert _gui_requested_engine_choice(explicit_cuda) == "cuda"


def test_failed_headless_install_does_not_persist_choice(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--install-root",
            str(tmp_path),
            "--runtime",
            "cpu",
            "--headless",
            "--install-engine",
            "--no-launch",
        ]
    )
    manager = CudaEngineManager(tmp_path, tmp_path / "app")
    with (
        patch("launcher.app.resolve_app_root", return_value=tmp_path / "app"),
        patch("launcher.app.CudaEngineManager", return_value=manager),
        patch.object(manager, "install", side_effect=LauncherError("installation failed")),
        pytest.raises(LauncherError, match="installation failed"),
    ):
        _run_engine_headless(args)

    assert not (tmp_path / "metadata" / "engine-choice.json").exists()


def test_successful_headless_validation_persists_actual_variant(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--install-root",
            str(tmp_path),
            "--runtime",
            "cpu",
            "--headless",
            "--install-engine",
            "--no-launch",
        ]
    )
    probe = EngineProbe("C:/private/python.exe", True, "ok", variant="cpu")
    manager = CudaEngineManager(tmp_path, tmp_path / "app")
    with (
        patch("launcher.app.resolve_app_root", return_value=tmp_path / "app"),
        patch("launcher.app.CudaEngineManager", return_value=manager),
        patch("launcher.app.probe_nvidia_device") as nvidia_probe,
        patch.object(manager, "install", return_value=probe),
    ):
        assert _run_engine_headless(args) == 0

    nvidia_probe.assert_not_called()
    payload = json.loads((tmp_path / "metadata" / "engine-choice.json").read_text(encoding="utf-8"))
    assert payload["variant"] == "cpu"


def test_cpu_and_cuda_engine_records_can_coexist(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    manager = CudaEngineManager(tmp_path, app)
    manager.save(EngineProbe("C:/cpu/python.exe", True, "ok", variant="cpu"), source="managed")
    manager.save(EngineProbe("C:/cuda/python.exe", True, "ok", variant="cuda"), source="managed")
    assert (tmp_path / "metadata" / "engine-cpu.json").is_file()
    assert (tmp_path / "metadata" / "engine-cuda.json").is_file()
    active = json.loads((tmp_path / "metadata" / "engine.json").read_text(encoding="utf-8"))
    assert active["variant"] == "cuda"


def test_dependency_install_keeps_cuda_torch_pinned(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    uv = tmp_path / "uv.exe"
    uv.touch()
    exact_python = (
        tmp_path
        / "engine"
        / "python"
        / "cpython-3.12.13-windows-x86_64-none"
        / "python.exe"
    )
    manager = CudaEngineManager(tmp_path, app)
    commands: list[list[str]] = []

    def capture(command, _progress, **_kwargs):
        commands.append(list(command))

    probe = EngineProbe(
        str(tmp_path / "engine" / "Scripts" / "python.exe"),
        True,
        "ok",
        torch_version="2.9.1+cu128",
        cuda_version="12.8",
        variant="cuda",
    )
    with (
        patch.object(manager, "locate_uv", return_value=uv),
        patch.object(
            manager,
            "_find_exact_managed_python",
            side_effect=(None, exact_python),
        ),
        patch.object(manager, "_validate_private_venv"),
        patch.object(manager, "_run_install_command", side_effect=capture),
        patch("launcher.engine.probe_engine", return_value=probe),
    ):
        manager.install(NvidiaDevice("RTX 5070 Ti", "610.62", (12, 0)), variant="cuda")

    download_command = commands[0]
    venv_command = commands[1]
    dependency_command = commands[3]
    assert download_command[1:3] == ["python", "install"]
    assert "3.12.13" in download_command
    assert "--managed-python" in download_command
    assert "--no-bin" in download_command
    assert "--no-registry" in download_command
    assert venv_command[0] == str(exact_python)
    assert venv_command[1:4] == ["-I", "-m", "venv"]
    assert "--copies" in venv_command
    assert "--without-pip" in venv_command
    assert "torch==2.9.1+cu128" in dependency_command
    assert "https://download.pytorch.org/whl/cu128" in dependency_command
    assert "https://pypi.org/simple" in dependency_command
    assert "unsafe-best-match" in dependency_command


def test_existing_full_version_python_avoids_uv_minor_version_link(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    uv = tmp_path / "uv.exe"
    uv.touch()
    exact_python = (
        tmp_path
        / "engine"
        / "python"
        / "cpython-3.12.13-windows-x86_64-none"
        / "python.exe"
    )
    exact_python.parent.mkdir(parents=True)
    exact_python.touch()
    manager = CudaEngineManager(tmp_path, app)
    commands: list[list[str]] = []

    def capture(command, _progress, **_kwargs):
        commands.append(list(command))

    probe = EngineProbe("C:/managed/python.exe", True, "ok", variant="cpu")
    version_result = subprocess.CompletedProcess([], 0, "3.12.13\n", "")
    with (
        patch.object(manager, "locate_uv", return_value=uv),
        patch("launcher.engine.subprocess.run", return_value=version_result),
        patch.object(manager, "_validate_private_venv"),
        patch.object(manager, "_run_install_command", side_effect=capture),
        patch("launcher.engine.probe_engine", return_value=probe),
    ):
        manager.install(None, variant="cpu")

    venv_command = commands[0]
    assert venv_command[0] == str(exact_python.resolve())
    assert venv_command[1:4] == ["-I", "-m", "venv"]
    assert "--copies" in venv_command
    assert "--without-pip" in venv_command
    assert all(command[1:3] != ["python", "install"] for command in commands)


def test_error_448_recovers_with_downloaded_exact_python(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    uv = tmp_path / "uv.exe"
    uv.touch()
    exact_python = (
        tmp_path
        / "engine"
        / "python"
        / "cpython-3.12.13-windows-x86_64-none"
        / "python.exe"
    )
    manager = CudaEngineManager(tmp_path, app)
    commands: list[list[str]] = []

    def fail_once_then_capture(command, _progress, **_kwargs):
        commands.append(list(command))
        if len(commands) == 1:
            exact_python.parent.mkdir(parents=True)
            exact_python.touch()
            raise LauncherError("无法遍历该路径，因为它包含不受信任的装入点。 (os error 448)")

    probe = EngineProbe("C:/managed/python.exe", True, "ok", variant="cpu")
    version_result = subprocess.CompletedProcess([], 0, "3.12.13\n", "")
    with (
        patch.object(manager, "locate_uv", return_value=uv),
        patch("launcher.engine.subprocess.run", return_value=version_result),
        patch.object(manager, "_validate_private_venv"),
        patch.object(manager, "_run_install_command", side_effect=fail_once_then_capture),
        patch("launcher.engine.probe_engine", return_value=probe),
    ):
        manager.install(None, variant="cpu")

    assert commands[0][1:3] == ["python", "install"]
    assert "--managed-python" in commands[0]
    recovery_command = commands[1]
    assert recovery_command[0] == str(exact_python.resolve())
    assert recovery_command[1:4] == ["-I", "-m", "venv"]
    assert "--copies" in recovery_command
    assert "--without-pip" in recovery_command


def test_error_448_without_usable_exact_python_still_fails(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    uv = tmp_path / "uv.exe"
    uv.touch()
    manager = CudaEngineManager(tmp_path, app)
    commands: list[list[str]] = []

    def fail(command, _progress, **_kwargs):
        commands.append(list(command))
        raise LauncherError("untrusted mount point (os error 448)")

    with (
        patch.object(manager, "locate_uv", return_value=uv),
        patch.object(manager, "_run_install_command", side_effect=fail),
        pytest.raises(LauncherError, match="仍未找到可执行的 Python"),
    ):
        manager.install(None, variant="cpu")

    assert len(commands) == 1


def test_non_448_python_creation_failure_does_not_retry(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    uv = tmp_path / "uv.exe"
    uv.touch()
    manager = CudaEngineManager(tmp_path, app)
    commands: list[list[str]] = []

    def fail(command, _progress, **_kwargs):
        commands.append(list(command))
        raise LauncherError("network interrupted")

    with (
        patch.object(manager, "locate_uv", return_value=uv),
        patch.object(manager, "_run_install_command", side_effect=fail),
        pytest.raises(LauncherError, match="network interrupted"),
    ):
        manager.install(None, variant="cpu")

    assert len(commands) == 1


def test_private_venv_validation_rejects_minor_version_junction_home(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    manager = CudaEngineManager(tmp_path, app)
    exact_python = (
        tmp_path
        / "engine"
        / "python"
        / "cpython-3.12.13-windows-x86_64-none"
        / "python.exe"
    )
    exact_python.parent.mkdir(parents=True)
    exact_python.touch()
    target = tmp_path / "engine" / "releases" / "cpu"
    python = target / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    config = target / "pyvenv.cfg"
    config.write_text(f"home = {exact_python.parent}\nversion = 3.12.13\n", encoding="utf-8")
    identity = {
        "version": "3.12.13",
        "base_prefix": str(exact_python.parent),
        "base_executable": str(exact_python),
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(identity) + "\n", "")

    with patch("launcher.engine.subprocess.run", return_value=completed):
        manager._validate_private_venv(python, exact_python)

    minor_home = exact_python.parent.with_name("cpython-3.12-windows-x86_64-none")
    config.write_text(f"home = {minor_home}\nversion = 3.12.13\n", encoding="utf-8")
    with pytest.raises(LauncherError, match="minor-version Junction"):
        manager._validate_private_venv(python, exact_python)


def test_installer_environment_ignores_host_python_conda_and_hermes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = tmp_path / "SPImaging" / "tools" / "uv.exe"
    uv.parent.mkdir(parents=True)
    uv.touch()
    hermes = tmp_path / "hermes" / "hermes-agent" / "venv" / "Scripts"
    conda = tmp_path / "miniconda3" / "Scripts"
    codex = tmp_path / "codex-runtimes" / "dependencies" / "bin"
    monkeypatch.setenv("PATH", os.pathsep.join((str(hermes), str(conda), str(codex))))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "host-python"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow-modules"))
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "user-site"))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "host-venv"))
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "host-conda"))
    monkeypatch.setenv("UV_PYTHON", str(tmp_path / "host-python.exe"))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "host-conda" / "ssl" / "certs"))
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "host-conda" / "ssl" / "cacert.pem"))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "organization-ca.pem"))

    private_python = tmp_path / "private-python"
    environment = _install_environment(
        uv,
        {
            "UV_PYTHON_INSTALL_DIR": str(private_python),
            "UV_CACHE_DIR": str(tmp_path / "cache"),
        },
    )

    clean_path = environment["PATH"].lower()
    assert str(uv.parent).lower() in clean_path
    assert "hermes" not in clean_path
    assert "conda" not in clean_path
    assert "codex" not in clean_path
    assert environment["UV_MANAGED_PYTHON"] == "1"
    assert environment["UV_PYTHON_INSTALL_DIR"] == str(private_python)
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert "PYTHONUSERBASE" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert "CONDA_PREFIX" not in environment
    assert "UV_PYTHON" not in environment
    assert "SSL_CERT_DIR" not in environment
    assert "SSL_CERT_FILE" not in environment
    assert environment["REQUESTS_CA_BUNDLE"] == str(tmp_path / "organization-ca.pem")


def test_failed_attempt_directory_is_removed_but_cache_is_retained(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    uv = tmp_path / "uv.exe"
    uv.touch()
    manager = CudaEngineManager(tmp_path, app)
    events: list[EngineProgress] = []

    def create_then_fail(command, _progress, **_kwargs):
        Path(command[-1]).mkdir(parents=True, exist_ok=True)
        raise LauncherError("network interrupted")

    with (
        patch.object(manager, "locate_uv", return_value=uv),
        patch.object(
            manager,
            "_find_exact_managed_python",
            return_value=tmp_path / "engine" / "python" / "exact" / "python.exe",
        ),
        patch.object(manager, "_run_install_command", side_effect=create_then_fail),
        pytest.raises(LauncherError, match="network interrupted"),
    ):
        manager.install(
            NvidiaDevice("RTX 5070 Ti", "610.62", (12, 0)),
            variant="cuda",
            progress=events.append,
        )
    releases = tmp_path / "engine" / "releases"
    assert releases.is_dir()
    assert list(releases.iterdir()) == []
    assert (tmp_path / "engine" / "cache" / "cuda").is_dir()
    assert any("已清理本次失败环境" in event.message for event in events)


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
