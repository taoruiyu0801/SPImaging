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
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Iterable, Sequence

from .errors import LauncherError


ENGINE_SCHEMA_VERSION = 1
ENGINE_PYTHON_VERSION = "3.12.13"
MINIMUM_FREE_BYTES = 8 * 1024**3
MINIMUM_CPU_FREE_BYTES = 4 * 1024**3
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

CPU_PROFILE = CudaProfile(
    "torch-2.9.1-cpu",
    "torch==2.9.1",
    "https://download.pytorch.org/whl/cpu",
    "",
)


@dataclass(frozen=True)
class EngineProgress:
    """One user-visible engine discovery or installation event."""

    stage: str
    step: int
    total_steps: int
    percent: float
    message: str
    kind: str = "status"


EngineProgressCallback = Callable[[EngineProgress], None]


def _emit(
    callback: EngineProgressCallback | None,
    stage: str,
    step: int,
    total_steps: int,
    percent: float,
    message: str,
    *,
    kind: str = "status",
) -> None:
    if callback is not None:
        callback(EngineProgress(stage, step, total_steps, max(0.0, min(100.0, percent)), message, kind))


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
    variant: str = "cuda"


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


def _install_environment(
    executable: str | Path,
    environment_overrides: dict[str, str],
) -> dict[str, str]:
    """Build an installer environment isolated from the host Python stack.

    Desktop hosts may prepend short-lived Conda, venv, Codex or Hermes paths to
    ``PATH``.  Letting ``uv`` inspect those paths can fail before it reaches the
    managed interpreter, and can also make a supposedly private engine depend
    on software that disappears when the host application exits.
    """

    environment = os.environ.copy()
    conda_roots = tuple(
        Path(value).expanduser().resolve()
        for key, value in environment.items()
        if key.upper() in {"CONDA_PREFIX", "CONDA_ROOT", "CONDA_EXE"} and value
    )
    for key in tuple(environment):
        upper = key.upper()
        if upper.startswith(("CONDA_", "UV_")) or upper in {
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
            "_CE_CONDA",
            "_CE_M",
        }:
            environment.pop(key, None)

    for key in tuple(environment):
        if key.upper() not in {"SSL_CERT_DIR", "SSL_CERT_FILE"}:
            continue
        value = environment.get(key, "")
        try:
            certificate_path = Path(value).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if any(certificate_path.is_relative_to(root) for root in conda_roots):
            environment.pop(key, None)

    path_entries = [Path(executable).resolve().parent]
    if os.name == "nt":
        windows_root = Path(environment.get("SystemRoot") or environment.get("WINDIR") or r"C:\Windows")
        path_entries.extend(
            (
                windows_root / "System32",
                windows_root,
                windows_root / "System32" / "Wbem",
                windows_root / "System32" / "WindowsPowerShell" / "v1.0",
            )
        )
    else:
        path_entries.extend(Path(item) for item in os.defpath.split(os.pathsep) if item)

    clean_path: list[str] = []
    seen: set[str] = set()
    for entry in path_entries:
        shown = str(entry)
        key = os.path.normcase(os.path.normpath(shown))
        if key not in seen:
            clean_path.append(shown)
            seen.add(key)

    environment["PATH"] = os.pathsep.join(clean_path)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "UV_NO_PROGRESS": "1",
            "UV_LINK_MODE": "copy",
            "UV_MANAGED_PYTHON": "1",
            "UV_NO_CONFIG": "1",
        }
    )
    environment.update(environment_overrides)
    return environment


_PROBE_PROGRAM = r'''
import importlib.util, json, platform, sys
result = {"python_version": sys.version.split()[0]}
required = json.loads(sys.argv[1])
index = int(sys.argv[2])
variant = sys.argv[3]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("缺少模块：" + ", ".join(missing))
import torch
from PySide6 import QtWidgets

if variant == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False")
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"GPU 编号 {index} 不存在")
    torch.cuda.set_device(index)
    device = torch.device(f"cuda:{index}")
else:
    if torch.version.cuda:
        raise RuntimeError(
            f"检测到 CUDA 版 PyTorch（运行时 {torch.version.cuda}），CPU 模式要求独立的 CPU 套件"
        )
    device = torch.device("cpu")

# Elementwise kernels catch wheels that can enumerate a new GPU but contain no
# code for its architecture. Conv3d mirrors SPImaging's real model workload on
# both engine variants.
x = torch.arange(256, dtype=torch.float32, device=device).reshape(1, 1, 8, 8, 4)
weight = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device)
y = torch.nn.functional.conv3d(x, weight, padding=1)
value = float((y + 1).mean().item())

if variant == "cuda":
    torch.cuda.synchronize(index)
    free, total = torch.cuda.mem_get_info(index)
    capability = torch.cuda.get_device_capability(index)
    device_name = torch.cuda.get_device_name(index)
else:
    free = total = 0
    capability = (0, 0)
    device_name = platform.processor() or "CPU"

result.update({
    "variant": variant,
    "torch_version": str(torch.__version__),
    "cuda_version": str(torch.version.cuda or ""),
    "device_name": device_name,
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
    variant: str = "cuda",
    gpu_index: int = 0,
    timeout_seconds: int = 45,
) -> EngineProbe:
    if variant not in {"cpu", "cuda"}:
        raise ValueError("engine variant must be cpu or cuda")
    executable = Path(python).expanduser()
    try:
        executable = executable.resolve(strict=True)
    except OSError:
        return EngineProbe(str(executable), False, "Python 可执行文件不存在", variant=variant)
    command = [
        str(executable),
        "-c",
        _PROBE_PROGRAM,
        json.dumps(REQUIRED_MODULES),
        str(gpu_index),
        variant,
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
        return EngineProbe(str(executable), False, f"{variant.upper()} 自检无法运行：{exc}", variant=variant)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "没有诊断输出").strip().splitlines()
        return EngineProbe(
            str(executable),
            False,
            detail[-1] if detail else f"{variant.upper()} 自检失败",
            variant=variant,
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        capability = tuple(int(item) for item in payload.get("compute_capability", (0, 0)))
        if len(capability) != 2:
            raise ValueError("invalid compute capability")
        return EngineProbe(
            str(executable),
            True,
            "真实 CUDA 张量与 3D 卷积自检通过"
            if variant == "cuda"
            else "CPU 张量与 3D 卷积自检通过",
            str(payload.get("python_version", "")),
            str(payload.get("torch_version", "")),
            str(payload.get("cuda_version", "")),
            str(payload.get("device_name", "")),
            (capability[0], capability[1]),
            int(payload.get("free_memory", 0)),
            int(payload.get("total_memory", 0)),
            str(payload.get("variant", variant)),
        )
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return EngineProbe(
            str(executable), False, f"{variant.upper()} 自检输出无法解析：{exc}", variant=variant
        )


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
    variant: str = "cuda",
) -> list[Path]:
    if variant not in {"cpu", "cuda"}:
        raise ValueError("engine variant must be cpu or cuda")
    root = Path(install_root).expanduser().resolve()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    environment = os.environ.get("SPIMAGING_ENGINE_PYTHON", "").strip()
    if environment:
        candidates.append(Path(environment))
    for record_name in (f"engine-{variant}.json", "engine.json"):
        recorded = _python_from_record(root / "metadata" / record_name)
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


def _directory_size(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for current, _directories, files in os.walk(root):
        for name in files:
            try:
                total += (Path(current) / name).stat().st_size
            except OSError:
                continue
    return total


class EngineManager:
    def __init__(self, install_root: str | Path, app_root: str | Path) -> None:
        self.install_root = Path(install_root).expanduser().resolve()
        self.app_root = Path(app_root).expanduser().resolve()
        self.record_path = self.install_root / "metadata" / "engine.json"

    def find(
        self,
        *,
        explicit: str | Path | None = None,
        variant: str = "cuda",
        gpu_index: int = 0,
        progress: EngineProgressCallback | None = None,
    ) -> tuple[EngineProbe | None, list[EngineProbe]]:
        if variant not in {"cpu", "cuda"}:
            raise ValueError("engine variant must be cpu or cuda")
        attempts: list[EngineProbe] = []
        candidates = discover_python_candidates(self.install_root, explicit=explicit, variant=variant)
        label = "CPU" if variant == "cpu" else "NVIDIA GPU"
        _emit(progress, "discover", 2, 6, 8, f"发现 {len(candidates)} 个 Python 环境，开始检查 {label} 兼容性")
        for index, candidate in enumerate(candidates, start=1):
            _emit(
                progress,
                "discover",
                2,
                6,
                8 + (7 * index / max(1, len(candidates))),
                f"[{index}/{len(candidates)}] 正在验证 {candidate}",
            )
            result = probe_engine(candidate, self.app_root, variant=variant, gpu_index=gpu_index)
            attempts.append(result)
            if result.compatible:
                _emit(progress, "discover", 2, 6, 15, f"环境通过自检：{candidate}", kind="log")
                self.save(result, source="existing")
                return result, attempts
            _emit(progress, "discover", 2, 6, 15, f"环境不兼容：{candidate}；{result.reason}", kind="log")
        return None, attempts

    def save(self, probe: EngineProbe, *, source: str, profile: str = "") -> None:
        payload = {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "source": source,
            "profile": profile,
            "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **asdict(probe),
        }
        for path in (self.record_path.with_name(f"engine-{probe.variant}.json"), self.record_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=".engine.", suffix=".tmp", dir=path.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
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

    @staticmethod
    def _discard_failed_target(
        target: Path,
        releases_root: Path,
        progress: EngineProgressCallback | None,
    ) -> None:
        """Remove only the directory created by the current failed attempt."""

        try:
            resolved_target = target.resolve()
            resolved_releases = releases_root.resolve()
            if resolved_target.parent != resolved_releases:
                raise LauncherError(f"拒绝清理预期目录之外的失败环境：{resolved_target}")
            if resolved_target.exists():
                shutil.rmtree(resolved_target)
                _emit(
                    progress,
                    "cleanup",
                    6,
                    6,
                    100,
                    f"已清理本次失败环境；下载缓存保留，可用于重试：{resolved_target.name}",
                    kind="log",
                )
        except OSError as exc:
            _emit(
                progress,
                "cleanup",
                6,
                6,
                100,
                f"无法清理本次失败环境 {target}：{exc}",
                kind="log",
            )

    def _run_install_command(
        self,
        command: Sequence[str],
        progress: EngineProgressCallback | None,
        *,
        stage: str,
        step: int,
        total_steps: int,
        start_percent: float,
        end_percent: float,
        label: str,
        activity_roots: Sequence[Path],
        environment_overrides: dict[str, str],
    ) -> None:
        environment = _install_environment(command[0], environment_overrides)
        _emit(progress, stage, step, total_steps, start_percent, label)
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

        output: queue.Queue[str | None] = queue.Queue()
        lines: list[str] = []
        assert process.stdout is not None

        def read_output() -> None:
            try:
                for raw in process.stdout:
                    line = raw.rstrip("\r\n")
                    if line:
                        output.put(line)
            finally:
                output.put(None)

        reader = threading.Thread(target=read_output, name="spimaging-engine-log", daemon=True)
        reader.start()
        started = time.monotonic()
        last_heartbeat = started
        last_measurement = started
        baseline_size = sum(_directory_size(path) for path in activity_roots)
        previous_size = baseline_size
        reader_done = False
        while not reader_done or process.poll() is None or not output.empty():
            try:
                item = output.get(timeout=0.25)
            except queue.Empty:
                item = ""
            if item is None:
                reader_done = True
            elif item:
                lines.append(item)
                _emit(progress, stage, step, total_steps, start_percent, item, kind="log")
            now = time.monotonic()
            if now - last_heartbeat >= 1.0 and process.poll() is None:
                elapsed = int(now - started)
                activity = f"{label}仍在进行，已用 {elapsed // 60:02d}:{elapsed % 60:02d}"
                if now - last_measurement >= 2.0:
                    current_size = sum(_directory_size(path) for path in activity_roots)
                    delta = max(0, current_size - previous_size)
                    total_delta = max(0, current_size - baseline_size)
                    speed = delta / max(0.1, now - last_measurement)
                    activity += f"；本次新增/缓存 {total_delta / 1024**2:.1f} MiB"
                    if speed > 0:
                        activity += f"；最近约 {speed / 1024**2:.1f} MiB/s"
                    previous_size = current_size
                    last_measurement = now
                _emit(progress, stage, step, total_steps, start_percent, activity, kind="activity")
                last_heartbeat = now
        returncode = process.wait()
        reader.join(timeout=2)
        if returncode != 0:
            detail = lines[-1] if lines else f"exit code {returncode}"
            raise LauncherError(f"{label}失败：{detail}")
        _emit(progress, stage, step, total_steps, end_percent, f"{label}完成", kind="log")

    def install(
        self,
        device: NvidiaDevice | None,
        *,
        variant: str = "cuda",
        gpu_index: int = 0,
        progress: EngineProgressCallback | None = None,
    ) -> EngineProbe:
        if variant not in {"cpu", "cuda"}:
            raise ValueError("engine variant must be cpu or cuda")
        if variant == "cuda":
            if device is None:
                raise LauncherError("GPU 模式需要先检测 NVIDIA 显卡和驱动")
            profile = select_cuda_profile(device)
        else:
            profile = CPU_PROFILE
        engine_root = self.install_root / "engine"
        engine_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(engine_root).free
        minimum_free = MINIMUM_FREE_BYTES if variant == "cuda" else MINIMUM_CPU_FREE_BYTES
        if free < minimum_free:
            raise LauncherError(
                f"{variant.upper()} 计算引擎至少需要 {minimum_free / 1024**3:.0f} GiB 可用空间，"
                f"当前只有 {free / 1024**3:.1f} GiB。"
            )
        _emit(
            progress,
            "preflight",
            1,
            6,
            4,
            f"磁盘预检通过：可用 {free / 1024**3:.1f} GiB；配置 {profile.profile_id}",
        )
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        target = engine_root / "releases" / f"{profile.profile_id}-py312-{stamp}"
        target.parent.mkdir(parents=True, exist_ok=True)
        releases_root = target.parent
        cache_root = engine_root / "cache" / variant
        python_root = engine_root / "python"
        cache_root.mkdir(parents=True, exist_ok=True)
        python_root.mkdir(parents=True, exist_ok=True)
        environment_overrides = {
            "UV_CACHE_DIR": str(cache_root),
            "UV_PYTHON_INSTALL_DIR": str(python_root),
        }
        uv = self.locate_uv()

        def run_step(command: Sequence[str], **kwargs: object) -> None:
            try:
                self._run_install_command(command, progress, **kwargs)  # type: ignore[arg-type]
            except Exception:
                self._discard_failed_target(target, releases_root, progress)
                raise

        run_step(
            [
                str(uv),
                "venv",
                "--python",
                ENGINE_PYTHON_VERSION,
                "--managed-python",
                str(target),
            ],
            stage="python",
            step=2,
            total_steps=6,
            start_percent=5,
            end_percent=15,
            label=f"创建 Python {ENGINE_PYTHON_VERSION} 私有环境",
            activity_roots=(target, cache_root, python_root),
            environment_overrides=environment_overrides,
        )
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run_step(
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
            stage="pytorch",
            step=3,
            total_steps=6,
            start_percent=16,
            end_percent=65,
            label=f"下载并安装 PyTorch {variant.upper()} 套件",
            activity_roots=(target, cache_root),
            environment_overrides=environment_overrides,
        )
        run_step(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(python),
                profile.torch_requirement,
                *BASE_REQUIREMENTS,
                "--index-url",
                profile.index_url,
                "--extra-index-url",
                "https://pypi.org/simple",
                "--index-strategy",
                "unsafe-best-match",
            ],
            stage="dependencies",
            step=4,
            total_steps=6,
            start_percent=66,
            end_percent=90,
            label="安装 SPImaging 图形界面和科学计算依赖",
            activity_roots=(target, cache_root),
            environment_overrides=environment_overrides,
        )
        test_label = "真实 CUDA 张量与 3D 卷积" if variant == "cuda" else "CPU 张量与 3D 卷积"
        _emit(progress, "self-test", 5, 6, 92, f"正在执行{test_label}自检")
        result = probe_engine(
            python,
            self.app_root,
            variant=variant,
            gpu_index=gpu_index,
            timeout_seconds=120,
        )
        if not result.compatible:
            self._discard_failed_target(target, releases_root, progress)
            raise LauncherError(f"新计算引擎未通过 {variant.upper()} 自检：{result.reason}")
        self.save(result, source="managed", profile=profile.profile_id)
        _emit(progress, "complete", 6, 6, 100, f"{variant.upper()} 计算引擎安装并验证完成")
        return result


# Kept as a public alias for the previous beta API and third-party scripts.
CudaEngineManager = EngineManager


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
    environment["SPIMAGING_CUDA_REQUIRED"] = "1" if probe.variant == "cuda" else "0"
    environment["SPIMAGING_RUNTIME_VARIANT"] = probe.variant
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
    "CPU_PROFILE",
    "CUDA_PROFILES",
    "CudaEngineManager",
    "CudaProfile",
    "EngineManager",
    "EngineProbe",
    "EngineProgress",
    "NvidiaDevice",
    "discover_python_candidates",
    "engine_environment",
    "launch_desktop_with_engine",
    "probe_engine",
    "probe_nvidia_device",
    "resolve_app_root",
    "select_cuda_profile",
]
