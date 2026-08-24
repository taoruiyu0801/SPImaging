"""Discovery, validation and optional installation of a CUDA compute engine.

The Windows installer deliberately does not contain Conda or PyTorch.  This
module first reuses a compatible user Python and only creates a small managed
``uv`` environment when no existing interpreter can execute a real CUDA
workload.  Merely trusting ``torch.cuda.is_available()`` is insufficient: an
old wheel may see a new GPU while containing no kernels for its architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Iterable, Sequence

from .errors import LauncherError


ENGINE_SCHEMA_VERSION = 1
ENGINE_PYTHON_VERSION = "3.12"
MINIMUM_FREE_BYTES = 8 * 1024**3
REQUIRED_MODULES = (
    "PySide6",
    "numpy",
    "h5py",
    "imageio",
    "skimage",
    "matplotlib",
    "tqdm",
    "cv2",
    "deepinv",
)
BASE_REQUIREMENTS = (
    "numpy<2",
    "h5py",
    "imageio",
    "scikit-image",
    "matplotlib",
    "tqdm",
    "opencv-python",
    "PySide6>=6.8,<7",
    "deepinv",
)


@dataclass(frozen=True)
class NvidiaDevice:
    name: str
    driver_version: str
    compute_capability: tuple[int, int]


@dataclass(frozen=True)
class CudaProfile:
    profile_id: str
    torch_requirement: str
    index_url: str
    minimum_driver: str
    minimum_compute: tuple[int, int] = (5, 0)


CUDA_PROFILES = (
    CudaProfile(
        "torch-2.9.1-cu128",
        "torch==2.9.1+cu128",
        "https://download.pytorch.org/whl/cu128",
        "570.65",
    ),
    CudaProfile(
        "torch-2.7.1-cu126",
        "torch==2.7.1+cu126",
        "https://download.pytorch.org/whl/cu126",
        "560.94",
    ),
)


@dataclass(frozen=True)
class EngineProbe:
    python: str
    compatible: bool
    reason: str
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    device_name: str = ""
    compute_capability: tuple[int, int] = (0, 0)
    free_memory: int = 0
    total_memory: int = 0


def _creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", str(value))
    return tuple(int(item) for item in numbers[:4]) or (0,)


def probe_nvidia_device(timeout_seconds: int = 8) -> NvidiaDevice:
    executable = shutil.which("nvidia-smi")
    if executable is None and os.name == "nt":
        common = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        executable = str(common) if common.is_file() else None
    if executable is None:
        raise LauncherError("未找到 nvidia-smi。SPImaging CUDA 版需要 NVIDIA 显卡和已安装的驱动。")
    command = [
        executable,
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            creationflags=_creationflags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LauncherError(f"NVIDIA 驱动检测失败：{exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "未知错误").strip().splitlines()[0]
        raise LauncherError(f"nvidia-smi 检测失败：{detail}")
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    fields = [item.strip() for item in line.split(",")]
    if len(fields) != 3:
        raise LauncherError(f"无法识别 nvidia-smi 返回内容：{line or '空'}")
    match = re.fullmatch(r"(\d+)\.(\d+)", fields[2])
    if match is None:
        raise LauncherError(f"无法识别 GPU 计算能力：{fields[2]}")
    return NvidiaDevice(fields[0], fields[1], (int(match.group(1)), int(match.group(2))))


def select_cuda_profile(device: NvidiaDevice) -> CudaProfile:
    if device.compute_capability < (5, 0):
        raise LauncherError(
            f"GPU {device.name} 的计算能力 {device.compute_capability[0]}.{device.compute_capability[1]} 不受当前版本支持。"
        )
    # Blackwell (sm_120) requires a CUDA 12.8-or-newer build.  Older cards may
    # use the slightly more permissive cu126 profile when their driver cannot
    # load cu128.
    candidates = CUDA_PROFILES[:1] if device.compute_capability >= (12, 0) else CUDA_PROFILES
    for profile in candidates:
        if _version_tuple(device.driver_version) >= _version_tuple(profile.minimum_driver):
            return profile
    minimum = min((profile.minimum_driver for profile in candidates), key=_version_tuple)
    raise LauncherError(
        f"NVIDIA 驱动 {device.driver_version} 过旧；{device.name} 至少需要驱动 {minimum}。"
        "请先从 NVIDIA 官方渠道更新驱动，SPImaging 不会自动修改系统驱动。"
    )


def resolve_app_root(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    environment = os.environ.get("SPIMAGING_APP_ROOT", "").strip()
    if environment:
        candidates.append(Path(environment))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "app")
    candidates.append(Path(__file__).resolve().parents[1])
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "spimaging" / "__init__.py").is_file():
            return root
    shown = ", ".join(str(item) for item in candidates)
    raise LauncherError(f"找不到 SPImaging 应用代码目录：{shown}")


def _probe_environment(app_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        upper = key.upper()
        if upper.startswith("CONDA_") or upper in {
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
            "_CE_CONDA",
            "_CE_M",
        }:
            environment.pop(key, None)
    environment["PYTHONPATH"] = str(app_root)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    return environment


_PROBE_PROGRAM = r'''
import importlib.util, json, sys
result = {"python_version": sys.version.split()[0]}
required = json.loads(sys.argv[1])
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("缺少模块：" + ", ".join(missing))
import torch
from PySide6 import QtWidgets
index = int(sys.argv[2])
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() 为 False")
if index >= torch.cuda.device_count():
    raise RuntimeError(f"GPU 编号 {index} 不存在")
torch.cuda.set_device(index)
device = torch.device(f"cuda:{index}")
# Elementwise kernels catch wheels that can enumerate a new GPU but contain no
# code for its architecture.  Conv3d mirrors SPImaging's real model workload.
x = torch.arange(256, dtype=torch.float32, device=device).reshape(1, 1, 8, 8, 4)
weight = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device)
y = torch.nn.functional.conv3d(x, weight, padding=1)
value = float((y + 1).mean().item())
torch.cuda.synchronize(index)
free, total = torch.cuda.mem_get_info(index)
capability = torch.cuda.get_device_capability(index)
result.update({
    "torch_version": str(torch.__version__),
    "cuda_version": str(torch.version.cuda or ""),
    "device_name": torch.cuda.get_device_name(index),
    "compute_capability": list(capability),
    "free_memory": int(free),
    "total_memory": int(total),
    "workload_value": value,
})
import spimaging
print(json.dumps(result, ensure_ascii=False))
'''


def probe_engine(
    python: str | Path,
    app_root: str | Path,
    *,
    gpu_index: int = 0,
    timeout_seconds: int = 45,
) -> EngineProbe:
    executable = Path(python).expanduser()
    try:
        executable = executable.resolve(strict=True)
    except OSError:
        return EngineProbe(str(executable), False, "Python 可执行文件不存在")
    command = [
        str(executable),
        "-c",
        _PROBE_PROGRAM,
        json.dumps(REQUIRED_MODULES),
        str(gpu_index),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path(app_root).expanduser().resolve(),
            env=_probe_environment(Path(app_root).expanduser().resolve()),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            creationflags=_creationflags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return EngineProbe(str(executable), False, f"CUDA 自检无法运行：{exc}")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "没有诊断输出").strip().splitlines()
        return EngineProbe(str(executable), False, detail[-1] if detail else "CUDA 自检失败")
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        capability = tuple(int(item) for item in payload.get("compute_capability", (0, 0)))
        if len(capability) != 2:
            raise ValueError("invalid compute capability")
        return EngineProbe(
            str(executable),
            True,
            "真实 CUDA 张量与 3D 卷积自检通过",
            str(payload.get("python_version", "")),
            str(payload.get("torch_version", "")),
            str(payload.get("cuda_version", "")),
            str(payload.get("device_name", "")),
            (capability[0], capability[1]),
            int(payload.get("free_memory", 0)),
            int(payload.get("total_memory", 0)),
        )
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return EngineProbe(str(executable), False, f"CUDA 自检输出无法解析：{exc}")


def _python_from_record(path: Path) -> Path | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    value = raw.get("python") if isinstance(raw, dict) else None
    return Path(value) if isinstance(value, str) and value.strip() else None


def _deduplicate(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key in seen or not resolved.is_file():
            continue
        if "windowsapps" in key:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def discover_python_candidates(
    install_root: str | Path,
    *,
    explicit: str | Path | None = None,
) -> list[Path]:
    root = Path(install_root).expanduser().resolve()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    environment = os.environ.get("SPIMAGING_ENGINE_PYTHON", "").strip()
    if environment:
        candidates.append(Path(environment))
    recorded = _python_from_record(root / "metadata" / "engine.json")
    if recorded is not None:
        candidates.append(recorded)

    profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
    known_roots = (
        profile / "anaconda3" / "envs",
        profile / "miniconda3" / "envs",
        profile / ".conda" / "envs",
        root / "engine" / "releases",
    )
    named: list[Path] = []
    other: list[Path] = []
    for env_root in known_roots:
        if not env_root.is_dir():
            continue
        for python in env_root.glob("*/python.exe" if os.name == "nt" else "*/bin/python"):
            (named if "spimaging" in python.parent.name.casefold() else other).append(python)
    candidates.extend(sorted(named))
    candidates.extend(sorted(other))

    if not getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable))
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["py", "-0p"], capture_output=True, text=True, timeout=5, creationflags=_creationflags()
            )
            for line in completed.stdout.splitlines():
                match = re.search(r"([A-Za-z]:\\.*?python(?:\d+(?:\.\d+)?)?\.exe)\s*$", line.strip())
                if match:
                    candidates.append(Path(match.group(1)))
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            completed = subprocess.run(
                ["where.exe", "python"], capture_output=True, text=True, timeout=5, creationflags=_creationflags()
            )
            candidates.extend(Path(line.strip()) for line in completed.stdout.splitlines() if line.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        found = shutil.which("python3") or shutil.which("python")
        if found:
            candidates.append(Path(found))
    return _deduplicate(candidates)


class CudaEngineManager:
    def __init__(self, install_root: str | Path, app_root: str | Path) -> None:
        self.install_root = Path(install_root).expanduser().resolve()
        self.app_root = Path(app_root).expanduser().resolve()
        self.record_path = self.install_root / "metadata" / "engine.json"

    def find(
        self,
        *,
        explicit: str | Path | None = None,
        gpu_index: int = 0,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[EngineProbe | None, list[EngineProbe]]:
        attempts: list[EngineProbe] = []
        for candidate in discover_python_candidates(self.install_root, explicit=explicit):
            if progress:
                progress(f"正在验证 {candidate}")
            result = probe_engine(candidate, self.app_root, gpu_index=gpu_index)
            attempts.append(result)
            if result.compatible:
                self.save(result, source="existing")
                return result, attempts
        return None, attempts

    def save(self, probe: EngineProbe, *, source: str, profile: str = "") -> None:
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "source": source,
            "profile": profile,
            "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **asdict(probe),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".engine.", suffix=".tmp", dir=self.record_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.record_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def locate_uv(self) -> Path:
        candidates = []
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).resolve().parent / "tools" / "uv.exe")
        candidates.append(self.app_root.parent / "tools" / ("uv.exe" if os.name == "nt" else "uv"))
        found = shutil.which("uv")
        if found:
            candidates.append(Path(found))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise LauncherError("找不到 uv 安装器。请重新安装 SPImaging，或从 https://astral.sh/uv 安装 uv。")

    def _run_install_command(
        self,
        command: Sequence[str],
        progress: Callable[[str], None] | None,
    ) -> None:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["UV_NO_PROGRESS"] = "1"
        environment["UV_LINK_MODE"] = "copy"
        try:
            process = subprocess.Popen(
                list(command),
                cwd=self.install_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_creationflags(),
            )
        except OSError as exc:
            raise LauncherError(f"无法启动计算引擎安装：{exc}") from exc
        lines: list[str] = []
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.strip()
            if line:
                lines.append(line)
                if progress:
                    progress(line)
        returncode = process.wait()
        if returncode != 0:
            detail = lines[-1] if lines else f"exit code {returncode}"
            raise LauncherError(f"CUDA 计算引擎安装失败：{detail}")

    def install(
        self,
        device: NvidiaDevice,
        *,
        gpu_index: int = 0,
        progress: Callable[[str], None] | None = None,
    ) -> EngineProbe:
        profile = select_cuda_profile(device)
        engine_root = self.install_root / "engine"
        engine_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(engine_root).free
        if free < MINIMUM_FREE_BYTES:
            raise LauncherError(
                f"CUDA 计算引擎至少需要 8 GiB 可用空间，当前只有 {free / 1024**3:.1f} GiB。"
            )
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        target = engine_root / "releases" / f"{profile.profile_id}-py312-{stamp}"
        target.parent.mkdir(parents=True, exist_ok=True)
        uv = self.locate_uv()
        if progress:
            progress(f"创建 Python {ENGINE_PYTHON_VERSION} 轻量环境")
        self._run_install_command(
            [
                str(uv),
                "venv",
                "--python",
                ENGINE_PYTHON_VERSION,
                "--python-preference",
                "managed",
                str(target),
            ],
            progress,
        )
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self._run_install_command(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(python),
                profile.torch_requirement,
                "--index-url",
                profile.index_url,
            ],
            progress,
        )
        self._run_install_command(
            [str(uv), "pip", "install", "--python", str(python), *BASE_REQUIREMENTS],
            progress,
        )
        if progress:
            progress("正在执行真实 CUDA 张量与 3D 卷积自检")
        result = probe_engine(python, self.app_root, gpu_index=gpu_index, timeout_seconds=120)
        if not result.compatible:
            raise LauncherError(f"新计算引擎未通过 CUDA 自检：{result.reason}")
        self.save(result, source="managed", profile=profile.profile_id)
        return result


def engine_environment(probe: EngineProbe, app_root: str | Path) -> dict[str, str]:
    environment = _probe_environment(Path(app_root).expanduser().resolve())
    python = Path(probe.python).resolve()
    prefix = python.parent.parent if python.parent.name.casefold() in {"scripts", "bin"} else python.parent
    bins = [python.parent, prefix, prefix / "Scripts", prefix / "Library" / "bin"]
    environment["PATH"] = os.pathsep.join(
        [*(str(path) for path in bins if path.is_dir()), environment.get("PATH", "")]
    ).rstrip(os.pathsep)
    environment["SPIMAGING_ENGINE_PYTHON"] = str(python)
    environment["SPIMAGING_APP_ROOT"] = str(Path(app_root).resolve())
    environment["SPIMAGING_CUDA_REQUIRED"] = "1"
    environment["SPIMAGING_RUNTIME_VARIANT"] = "cuda"
    return environment


def launch_desktop_with_engine(
    probe: EngineProbe,
    app_root: str | Path,
    *,
    launcher_executable: str | Path | None = None,
) -> subprocess.Popen[bytes]:
    python = Path(probe.python).resolve()
    executable = python
    if os.name == "nt" and python.name.casefold() == "python.exe":
        windowed = python.with_name("pythonw.exe")
        if windowed.is_file():
            executable = windowed
    environment = engine_environment(probe, app_root)
    if launcher_executable:
        environment["SPIMAGING_LAUNCHER_EXE"] = str(Path(launcher_executable).resolve())
    return subprocess.Popen(
        [str(executable), "-m", "spimaging.desktop"],
        cwd=Path(app_root).resolve(),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_creationflags(),
    )


__all__ = [
    "BASE_REQUIREMENTS",
    "CUDA_PROFILES",
    "CudaEngineManager",
    "CudaProfile",
    "EngineProbe",
    "NvidiaDevice",
    "discover_python_candidates",
    "engine_environment",
    "launch_desktop_with_engine",
    "probe_engine",
    "probe_nvidia_device",
    "resolve_app_root",
    "select_cuda_profile",
]
