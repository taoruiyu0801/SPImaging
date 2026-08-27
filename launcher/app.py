"""Minimal Chinese bootstrap UI/CLI for the frozen ``SPImaging.exe``."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Sequence

from .activation import ActivationManager
from .bootstrap import ProvisionResult, Provisioner, launch_desktop
from .device import probe_nvidia
from .download import DownloadTransport, LocalDirectoryTransport, UrllibTransport, fetch_bytes
from .errors import LauncherError
from .engine import (
    CudaEngineManager,
    EngineProgress,
    EngineProbe,
    NvidiaDevice,
    launch_desktop_with_engine,
    probe_nvidia_device,
    resolve_app_root,
    select_cuda_profile,
)
from .locking import InterProcessLock
from .manifest import NvidiaCapability, ReleaseManifest, compare_semver, select_runtime_asset
from .signing import WindowsSignatureVerifier
from .update import ManifestCache


DEFAULT_BETA_MANIFEST_URL = (
    "https://github.com/ewellchen/SPImaging/releases/download/windows-beta/"
    "spimaging-release-manifest.json"
)
DEFAULT_STABLE_MANIFEST_URL = (
    "https://github.com/ewellchen/SPImaging/releases/latest/download/"
    "spimaging-release-manifest.json"
)


def default_install_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "SPImaging"
    return Path.home() / "AppData" / "Local" / "SPImaging"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPImaging private-runtime bootstrap launcher")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest-url")
    source.add_argument("--manifest-file", type=Path, help="validated local manifest for packaging tests")
    parser.add_argument("--asset-dir", type=Path, help="offline asset directory used with --manifest-file")
    parser.add_argument(
        "--channel",
        choices=("beta", "stable"),
        default="beta",
        help="更新通道；beta 使用显式 windows-beta 清单，不依赖 GitHub latest。",
    )
    parser.add_argument("--install-root", type=Path, default=default_install_root())
    parser.add_argument(
        "--runtime",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="运行时偏好；省略时读取桌面设置，默认 auto。",
    )
    parser.add_argument("--headless", action="store_true", help="do not create the bootstrap window")
    parser.add_argument("--dry-run", action="store_true", help="validate/select only; do not download or launch")
    parser.add_argument("--no-launch", action="store_true", help="provision but do not start the desktop")
    parser.add_argument("--repair", action="store_true", help="recheck and reinstall the selected release")
    parser.add_argument("--check-now", action="store_true", help="ignore the 24-hour update-check interval")
    parser.add_argument("--accept-update", action="store_true", help="allow update without an interactive confirmation")
    parser.add_argument("--wait-for-pid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--engine-python", type=Path, help="优先验证指定的 CUDA Python 环境")
    parser.add_argument("--app-root", type=Path, help="安装版应用代码目录")
    parser.add_argument("--gpu-index", type=int, default=0, help="必须通过自检的 NVIDIA GPU 编号")
    parser.add_argument("--install-engine", action="store_true", help="缺失时安装 CUDA 计算引擎")
    parser.add_argument("--repair-engine", action="store_true", help="重新安装并验证 CUDA 计算引擎")
    parser.add_argument("--legacy-runtime", action="store_true", help=argparse.SUPPRESS)
    return parser


@dataclass(frozen=True)
class ResolvedManifest:
    manifest: ReleaseManifest
    raw: bytes
    signature: bytes | None
    message: str


def load_desktop_preferences(install_root: Path) -> dict[str, object]:
    """Read the small desktop settings file without trusting arbitrary fields."""

    path = install_root.expanduser().resolve() / "settings.json"
    try:
        if path.stat().st_size > 1024 * 1024:
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, object] = {}
    if raw.get("device") in {"auto", "cuda", "cpu"}:
        result["device"] = raw["device"]
    if isinstance(raw.get("cache_dir"), str) and raw["cache_dir"].strip():
        result["cache_dir"] = raw["cache_dir"]
    if isinstance(raw.get("update_checks"), bool):
        result["update_checks"] = raw["update_checks"]
    return result


def runtime_preference(args: argparse.Namespace) -> str:
    if args.runtime is not None:
        return str(args.runtime)
    return str(load_desktop_preferences(args.install_root).get("device", "auto"))


def _updates_disabled(args: argparse.Namespace) -> bool:
    if args.manifest_file is not None or args.check_now or args.repair:
        return False
    return load_desktop_preferences(args.install_root).get("update_checks") is False


def _wait_for_pid(pid: int | None, timeout_seconds: int = 300) -> None:
    if pid is None or pid <= 0 or pid == os.getpid():
        return
    if os.name == "nt":
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
        if handle:
            try:
                ctypes.windll.kernel32.WaitForSingleObject(handle, timeout_seconds * 1000)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
            return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)


def _transport_for_args(args: argparse.Namespace) -> DownloadTransport:
    if args.asset_dir is not None:
        if args.manifest_file is None:
            raise LauncherError("--asset-dir requires --manifest-file")
        try:
            return LocalDirectoryTransport(args.asset_dir)
        except (OSError, ValueError) as error:
            raise LauncherError(f"无法打开离线资产目录：{error}") from error
    return UrllibTransport()


def resolve_manifest(args: argparse.Namespace, transport: DownloadTransport) -> ResolvedManifest:
    cache = ManifestCache(args.install_root)
    if args.manifest_file is not None:
        try:
            raw = args.manifest_file.read_bytes()
        except OSError as error:
            raise LauncherError(f"无法读取本地发布清单：{error}") from error
        manifest = ReleaseManifest.from_json(raw)
        if manifest.channel != args.channel:
            raise LauncherError(
                f"本地发布清单通道 {manifest.channel} 与请求的 {args.channel} 通道不一致"
            )
        signature = None
        if cache.trust_policy.signature_required:
            signature_path = args.manifest_file.with_name(args.manifest_file.name + ".p7s")
            try:
                signature = signature_path.read_bytes()
            except OSError as error:
                raise LauncherError(f"无法读取本地发布清单签名：{error}") from error
        cache.trust_policy.verify_manifest(raw, manifest, signature)
        cache.validate_progression(raw, manifest, expected_channel=args.channel)
        return ResolvedManifest(manifest, raw, signature, "已验证本地发布清单")
    manifest_url = args.manifest_url or (
        DEFAULT_BETA_MANIFEST_URL if args.channel == "beta" else DEFAULT_STABLE_MANIFEST_URL
    )
    manifest, _fetched, message = cache.resolve(
        manifest_url,
        transport,
        force_check=args.check_now,
        expected_channel=args.channel,
    )
    record = cache.load_record()
    if record is None or record[0] != manifest:
        raise LauncherError("无法读取已经验证的发布清单缓存")
    _cached_manifest, raw, signature = record
    return ResolvedManifest(manifest, raw, signature, message)


def _installed_result(install_root: Path, manifest: ReleaseManifest) -> ProvisionResult | None:
    manager = ActivationManager(install_root)
    runtime = manager.active_path("runtime")
    app = manager.active_path("app")
    runtime_record = manager.active_record("runtime") or {}
    app_record = manager.active_record("app") or {}
    if runtime is None or app is None:
        return None
    metadata = runtime_record.get("metadata") if isinstance(runtime_record.get("metadata"), dict) else {}
    variant = metadata.get("variant", "cpu")
    app_metadata = app_record.get("metadata") if isinstance(app_record.get("metadata"), dict) else {}
    version = app_metadata.get("version")
    if not isinstance(version, str):
        active_id = app_record.get("active")
        version = (
            active_id[: -len("-universal")]
            if isinstance(active_id, str) and active_id.endswith("-universal")
            else manifest.release_version
        )
    if not isinstance(variant, str):
        variant = "cpu"
    return ProvisionResult(version, runtime, app, variant, False, "使用已经安装并验证的运行时")


def _is_update(installed: ProvisionResult | None, manifest: ReleaseManifest) -> bool:
    return installed is not None and compare_semver(manifest.release_version, installed.release_version) > 0


def _is_downgrade(installed: ProvisionResult | None, manifest: ReleaseManifest) -> bool:
    return installed is not None and compare_semver(manifest.release_version, installed.release_version) < 0


def run_install(
    args: argparse.Namespace,
    resolved: ResolvedManifest,
    capability: NvidiaCapability,
    transport: DownloadTransport,
    progress: Callable[[str, int, int], None] | None = None,
) -> ProvisionResult:
    maintenance_path = args.install_root.expanduser().resolve() / "metadata" / "maintenance.lock"
    with InterProcessLock(maintenance_path, purpose="安装/更新维护"):
        preferences = load_desktop_preferences(args.install_root)
        cache_value = preferences.get("cache_dir")
        provisioner = Provisioner(
            args.install_root,
            transport,
            cache_root=Path(cache_value) if isinstance(cache_value, str) else None,
            verifier=WindowsSignatureVerifier(),
            progress=progress,
        )
        snapshot = provisioner.manager.snapshot_state()
        try:
            result = provisioner.provision(
                resolved.manifest,
                runtime_preference(args),
                capability,
                force=args.repair,
            )
            ManifestCache(args.install_root).mark_active(
                resolved.raw,
                resolved.manifest,
                resolved.signature,
            )
            return result
        except Exception:
            provisioner.manager.restore_state(snapshot)
            raise


def _launch_existing(args: argparse.Namespace) -> int:
    cache = ManifestCache(args.install_root)
    active_manifest = cache.load_active()
    if active_manifest is None:
        raise LauncherError("没有可离线启动的已验证版本")
    result = _installed_result(args.install_root, active_manifest)
    if result is None:
        raise LauncherError("已安装运行时不完整，请联网修复")
    if not args.no_launch:
        return int(launch_desktop(active_manifest, result).wait())
    return 0


def _run_headless(args: argparse.Namespace) -> int:
    if _updates_disabled(args):
        try:
            return _launch_existing(args)
        except LauncherError:
            pass
    transport = _transport_for_args(args)
    try:
        resolved = resolve_manifest(args, transport)
    except LauncherError:
        if not args.dry_run:
            return _launch_existing(args)
        raise
    capability = probe_nvidia()
    selection = select_runtime_asset(resolved.manifest, runtime_preference(args), capability)
    if args.dry_run:
        print(f"manifest={resolved.manifest.release_version}")
        print(f"runtime={selection.selected_variant}")
        print(f"reason={selection.reason}")
        return 0
    installed = _installed_result(args.install_root, resolved.manifest)
    if _is_downgrade(installed, resolved.manifest):
        return _launch_existing(args)
    if _is_update(installed, resolved.manifest) and not args.accept_update and not args.repair:
        return _launch_existing(args)
    result = run_install(args, resolved, capability, transport)
    print(result.reason)
    if not args.no_launch:
        return int(launch_desktop(resolved.manifest, result).wait())
    return 0


def _run_gui(args: argparse.Namespace) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as error:
        raise LauncherError("启动器图形组件不可用，请使用 --headless 查看诊断") from error

    root = tk.Tk()
    root.title("SPImaging 安装与启动")
    root.geometry("560x280")
    root.resizable(False, False)
    status = tk.StringVar(value="正在检查运行环境……")
    detail = tk.StringVar(value="首次启动会下载经过校验的私有运行环境。")
    progress_value = tk.DoubleVar(value=0)
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="SPImaging", font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w")
    ttk.Label(frame, textvariable=status, font=("Microsoft YaHei UI", 11)).pack(anchor="w", pady=(18, 4))
    ttk.Label(frame, textvariable=detail, wraplength=500).pack(anchor="w")
    bar = ttk.Progressbar(frame, variable=progress_value, maximum=100)
    bar.pack(fill="x", pady=(20, 12))
    close_button = ttk.Button(frame, text="关闭", command=root.destroy, state="disabled")
    close_button.pack(anchor="e")
    events: queue.Queue[tuple[str, object]] = queue.Queue()
    launched_process: subprocess.Popen[bytes] | None = None

    def progress(name: str, current: int, total: int) -> None:
        events.put(("progress", (name, current, total)))

    def work() -> None:
        transport = _transport_for_args(args)
        try:
            if _updates_disabled(args):
                cache = ManifestCache(args.install_root)
                active_manifest = cache.load_active()
                installed = _installed_result(args.install_root, active_manifest) if active_manifest else None
                if active_manifest is not None and installed is not None:
                    events.put(("status", "已按设置关闭自动更新检查，使用本地版本"))
                    events.put(("done", (active_manifest, installed)))
                    return
            try:
                resolved = resolve_manifest(args, transport)
            except LauncherError as fetch_error:
                cache = ManifestCache(args.install_root)
                active_manifest = cache.load_active()
                installed = _installed_result(args.install_root, active_manifest) if active_manifest else None
                if active_manifest is None or installed is None:
                    raise fetch_error
                events.put(("status", f"当前离线，启动已验证版本：{fetch_error}"))
                events.put(("done", (active_manifest, installed)))
                return
            events.put(("status", resolved.message))
            capability = probe_nvidia()
            installed = _installed_result(args.install_root, resolved.manifest)
            if _is_downgrade(installed, resolved.manifest):
                cache = ManifestCache(args.install_root)
                active_manifest = cache.load_active()
                active_result = _installed_result(args.install_root, active_manifest) if active_manifest else None
                if active_manifest is None or active_result is None:
                    raise LauncherError("拒绝安装低于当前版本的发布清单")
                events.put(("status", "已拒绝版本降级，继续启动当前版本"))
                events.put(("done", (active_manifest, active_result)))
                return
            if _is_update(installed, resolved.manifest) and not args.accept_update and not args.repair:
                events.put(("confirm", (resolved, capability, transport)))
                return
            result = run_install(args, resolved, capability, transport, progress)
            events.put(("done", (resolved.manifest, result)))
        except Exception as error:
            events.put(("error", error))

    def install_after_confirm(payload: tuple[ResolvedManifest, NvidiaCapability, DownloadTransport], accepted: bool) -> None:
        resolved, capability, transport = payload
        try:
            if accepted:
                result = run_install(args, resolved, capability, transport, progress)
                events.put(("done", (resolved.manifest, result)))
            else:
                cache = ManifestCache(args.install_root)
                active_manifest = cache.load_active()
                installed = _installed_result(args.install_root, active_manifest) if active_manifest else None
                if active_manifest is None or installed is None:
                    raise LauncherError("当前安装不完整，无法跳过更新")
                events.put(("done", (active_manifest, installed)))
        except Exception as error:
            events.put(("error", error))

    def poll() -> None:
        nonlocal launched_process
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "progress":
                    name, current, total = payload  # type: ignore[misc]
                    status.set(f"正在下载 {name}")
                    detail.set(f"{current / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MiB")
                    progress_value.set(current * 100 / total)
                elif kind == "status":
                    detail.set(str(payload))
                elif kind == "confirm":
                    resolved, _, _ = payload  # type: ignore[misc]
                    accepted = messagebox.askyesno(
                        "发现更新",
                        f"发现 SPImaging {resolved.manifest.release_version}。现在安装更新吗？",
                        parent=root,
                    )
                    threading.Thread(target=install_after_confirm, args=(payload, accepted), daemon=True).start()  # type: ignore[arg-type]
                elif kind == "done":
                    manifest, result = payload  # type: ignore[misc]
                    progress_value.set(100)
                    status.set("环境已就绪")
                    detail.set(result.reason)
                    if not args.no_launch:
                        launched_process = launch_desktop(manifest, result)
                        root.after(300, root.destroy)
                    else:
                        close_button.configure(state="normal")
                elif kind == "error":
                    status.set("启动失败")
                    detail.set(str(payload))
                    close_button.configure(state="normal")
                    messagebox.showerror("SPImaging 启动失败", str(payload), parent=root)
        except queue.Empty:
            pass
        if root.winfo_exists():
            root.after(100, poll)

    threading.Thread(target=work, daemon=True).start()
    root.after(100, poll)
    root.mainloop()
    if launched_process is not None:
        return int(launched_process.wait())
    return 0


def _launcher_executable() -> Path | None:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()
    return None


def _legacy_runtime_requested(args: argparse.Namespace) -> bool:
    """Keep the previous manifest runtime available only for old beta media."""

    return bool(
        args.legacy_runtime
        or args.manifest_file is not None
        or args.manifest_url is not None
        or args.asset_dir is not None
    )


def _engine_summary(probe: EngineProbe) -> str:
    if probe.variant == "cpu":
        return f"CPU · PyTorch {probe.torch_version} · Python {probe.python_version}"
    capability = f"sm_{probe.compute_capability[0]}{probe.compute_capability[1]}"
    memory = f"{probe.total_memory / 1024**3:.1f} GiB" if probe.total_memory else "未知显存"
    return (
        f"{probe.device_name} · PyTorch {probe.torch_version} · CUDA {probe.cuda_version} · "
        f"{capability} · {memory}"
    )


def _engine_choice_path(install_root: Path) -> Path:
    return install_root.expanduser().resolve() / "metadata" / "engine-choice.json"


def _load_engine_choice(args: argparse.Namespace) -> str | None:
    if args.runtime in {"cpu", "cuda"}:
        return str(args.runtime)
    for path in (
        _engine_choice_path(args.install_root),
        args.install_root.expanduser().resolve() / "metadata" / "engine.json",
    ):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(raw, dict):
            value = raw.get("variant")
            if value in {"cpu", "cuda"}:
                return str(value)
    preference = load_desktop_preferences(args.install_root).get("device")
    return str(preference) if preference in {"cpu", "cuda"} else None


def _gui_requested_engine_choice(args: argparse.Namespace) -> str | None:
    """Return only a choice explicitly supplied for this GUI invocation.

    A persisted choice describes the last successfully validated engine, but
    it must never turn an ordinary double-click into authorization to download
    or install software.  Explicit ``--runtime cpu``/``--runtime cuda`` calls
    are used by controlled repair and packaging flows and may start directly.
    """

    return str(args.runtime) if args.runtime in {"cpu", "cuda"} else None


def _save_engine_choice(install_root: Path, variant: str) -> None:
    if variant not in {"cpu", "cuda"}:
        raise ValueError("engine variant must be cpu or cuda")
    path = _engine_choice_path(install_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "schema_version": 1,
        "variant": variant,
        "selected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _headless_engine_logger(install_root: Path) -> tuple[Path, Callable[[EngineProgress], None]]:
    logs_root = install_root.expanduser().resolve() / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / f"engine-install-{datetime.now():%Y%m%d-%H%M%S}.log"

    def emit(event: EngineProgress) -> None:
        line = f"[{datetime.now():%H:%M:%S}] [{event.step}/{event.total_steps}] {event.message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    return path, emit


def _run_engine_headless(args: argparse.Namespace) -> int:
    app_root = resolve_app_root(args.app_root)
    manager = CudaEngineManager(args.install_root, app_root)
    log_path, report = _headless_engine_logger(args.install_root)
    print(f"安装日志：{log_path}", flush=True)
    variant = _load_engine_choice(args)
    if variant is None:
        try:
            device = probe_nvidia_device()
            select_cuda_profile(device)
            variant = "cuda"
        except LauncherError:
            device = None
            variant = "cpu"
    else:
        device = probe_nvidia_device() if variant == "cuda" else None
    if args.install_engine or args.repair_engine:
        probe = manager.install(
            device,
            variant=variant,
            gpu_index=args.gpu_index,
            progress=report,
        )
    else:
        probe, attempts = manager.find(
            explicit=args.engine_python,
            variant=variant,
            gpu_index=args.gpu_index,
            progress=report,
        )
        if probe is None:
            detail = attempts[-1].reason if attempts else "没有找到 Python 环境"
            profile = select_cuda_profile(device).profile_id if device is not None else "torch-2.9.1-cpu"
            raise LauncherError(
                f"没有可用的 {variant.upper()} 计算引擎（{detail}）。可安装配置：{profile}；"
                "请使用 --install-engine。"
            )
    _save_engine_choice(args.install_root, probe.variant)
    print(_engine_summary(probe))
    if not args.no_launch:
        process = launch_desktop_with_engine(
            probe,
            app_root,
            launcher_executable=_launcher_executable(),
        )
        return int(process.wait())
    return 0


def _run_engine_gui(args: argparse.Namespace) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as error:
        raise LauncherError("启动器图形组件不可用，请使用 --headless 查看诊断") from error

    app_root = resolve_app_root(args.app_root)
    manager = CudaEngineManager(args.install_root, app_root)
    root = tk.Tk()
    root.title("SPImaging 运行环境")
    root.geometry("780x650")
    root.minsize(720, 580)
    status = tk.StringVar(value="请选择运行方式")
    detail = tk.StringVar(value="CPU 适合无 NVIDIA 显卡的电脑；GPU 模式会先检查驱动和已有环境。")
    step_text = tk.StringVar(value="尚未开始")
    progress_value = tk.DoubleVar(value=0)
    events: queue.Queue[tuple[str, object]] = queue.Queue()
    launched_process: subprocess.Popen[bytes] | None = None
    selected_variant: str | None = None
    busy = False

    logs_root = args.install_root.expanduser().resolve() / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"engine-install-{datetime.now():%Y%m%d-%H%M%S}.log"
    log_lock = threading.Lock()

    frame = ttk.Frame(root, padding=22)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="SPImaging", font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w")
    ttk.Label(frame, text="选择并准备本机计算环境", font=("Microsoft YaHei UI", 10)).pack(anchor="w", pady=(2, 12))

    choice = ttk.Frame(frame)
    choice.pack(fill="x", pady=(0, 12))
    cpu_button = ttk.Button(choice, text="使用 CPU（下载 CPU 套件）", width=28)
    cpu_button.pack(side="left", padx=(0, 8))
    gpu_button = ttk.Button(choice, text="使用 NVIDIA GPU（检查并安装）", width=31)
    gpu_button.pack(side="left")

    ttk.Separator(frame).pack(fill="x", pady=(0, 12))
    ttk.Label(frame, textvariable=status, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
    ttk.Label(frame, textvariable=detail, wraplength=720).pack(anchor="w", pady=(4, 8))
    ttk.Label(frame, textvariable=step_text).pack(anchor="w")
    bar = ttk.Progressbar(frame, variable=progress_value, maximum=100, mode="determinate")
    bar.pack(fill="x", pady=(6, 12))

    log_header = ttk.Frame(frame)
    log_header.pack(fill="x")
    ttk.Label(log_header, text="安装日志（实时更新）", font=("Microsoft YaHei UI", 10, "bold")).pack(side="left")
    ttk.Label(log_header, text=str(log_path), foreground="#666666").pack(side="right")
    log_frame = ttk.Frame(frame)
    log_frame.pack(fill="both", expand=True, pady=(5, 10))
    log_widget = tk.Text(
        log_frame,
        height=15,
        wrap="word",
        state="disabled",
        font=("Consolas", 9),
        background="#111827",
        foreground="#e5e7eb",
        insertbackground="#e5e7eb",
    )
    scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log_widget.yview)
    log_widget.configure(yscrollcommand=scrollbar.set)
    log_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    actions = ttk.Frame(frame)
    actions.pack(fill="x")
    retry_button = ttk.Button(actions, text="重试当前模式", state="disabled")
    retry_button.pack(side="left")
    choose_button = ttk.Button(actions, text="重新选择 CPU/GPU", state="disabled")
    choose_button.pack(side="left", padx=(8, 0))
    copy_button = ttk.Button(actions, text="复制全部日志")
    copy_button.pack(side="right", padx=(8, 0))
    close_button = ttk.Button(actions, text="关闭", command=root.destroy)
    close_button.pack(side="right")

    def append_log(message: str) -> None:
        log_widget.configure(state="normal")
        log_widget.insert("end", message + "\n")
        log_widget.see("end")
        log_widget.configure(state="disabled")

    def record(message: str) -> str:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        with log_lock:
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        return line

    def emit_progress(event: EngineProgress) -> None:
        line = record(f"[{event.step}/{event.total_steps}] {event.message}")
        events.put(("progress", (event, line)))

    def set_choice_enabled(enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        cpu_button.configure(state=state)
        gpu_button.configure(state=state)

    def run_variant(variant: str, *, force: bool = False, explicit: bool = False) -> None:
        nonlocal selected_variant, busy
        if busy:
            return
        selected_variant = variant
        busy = True
        set_choice_enabled(False)
        retry_button.configure(state="disabled")
        choose_button.configure(state="disabled")
        close_button.configure(state="disabled")
        progress_value.set(1)
        label = "CPU" if variant == "cpu" else "NVIDIA GPU"
        status.set(f"正在准备 {label} 运行环境")
        detail.set("下面会持续显示每一步和完整日志；长时间下载时每秒都会显示已用时间。")
        choice_source = "命令行明确指定" if explicit else "用户选择"
        append_log(record(f"{choice_source} {label} 模式；应用目录：{app_root}"))

        def work() -> None:
            try:
                device: NvidiaDevice | None = None
                emit_progress(EngineProgress("hardware", 1, 6, 2, f"开始检查 {label} 模式"))
                if variant == "cuda":
                    emit_progress(EngineProgress("hardware", 1, 6, 3, "正在运行 nvidia-smi 检查显卡和驱动"))
                    device = probe_nvidia_device()
                    profile = select_cuda_profile(device)
                    emit_progress(
                        EngineProgress(
                            "hardware",
                            1,
                            6,
                            5,
                            f"检测到 {device.name}；驱动 {device.driver_version}；"
                            f"计算能力 {device.compute_capability[0]}.{device.compute_capability[1]}；"
                            f"匹配 {profile.profile_id}",
                        )
                    )
                else:
                    emit_progress(EngineProgress("hardware", 1, 6, 5, "CPU 模式无需 NVIDIA 显卡或 CUDA"))

                result: EngineProbe | None = None
                attempts: list[EngineProbe] = []
                if not force and not args.install_engine and not args.repair_engine:
                    result, attempts = manager.find(
                        explicit=args.engine_python,
                        variant=variant,
                        gpu_index=args.gpu_index,
                        progress=emit_progress,
                    )
                if result is None:
                    if attempts:
                        emit_progress(
                            EngineProgress(
                                "discover",
                                2,
                                6,
                                15,
                                f"已有环境均不兼容，最后一次原因：{attempts[-1].reason}",
                                "log",
                            )
                        )
                    emit_progress(
                        EngineProgress("install", 2, 6, 15, f"没有可复用的 {label} 环境，开始创建私有环境")
                    )
                    result = manager.install(
                        device,
                        variant=variant,
                        gpu_index=args.gpu_index,
                        progress=emit_progress,
                    )
                events.put(("done", result))
            except Exception as error:
                record(f"失败：{type(error).__name__}: {error}")
                events.put(("error", error))

        threading.Thread(target=work, name=f"spimaging-{variant}-setup", daemon=True).start()

    def reset_choice() -> None:
        nonlocal selected_variant, busy
        if busy:
            return
        selected_variant = None
        status.set("请选择运行方式")
        detail.set("CPU 适合无 NVIDIA 显卡的电脑；GPU 模式会先检查驱动和已有环境。")
        step_text.set("尚未开始")
        progress_value.set(0)
        retry_button.configure(state="disabled")
        choose_button.configure(state="disabled")
        set_choice_enabled(True)

    def copy_logs() -> None:
        try:
            content = log_path.read_text(encoding="utf-8")
        except OSError as error:
            messagebox.showerror("无法读取日志", str(error), parent=root)
            return
        root.clipboard_clear()
        root.clipboard_append(content)
        detail.set("完整安装日志已复制到剪贴板。")

    cpu_button.configure(command=lambda: run_variant("cpu"))
    gpu_button.configure(command=lambda: run_variant("cuda"))
    retry_button.configure(command=lambda: selected_variant and run_variant(selected_variant))
    choose_button.configure(command=reset_choice)
    copy_button.configure(command=copy_logs)

    def poll() -> None:
        nonlocal launched_process, busy
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "progress":
                    event, line = payload  # type: ignore[misc]
                    progress_value.set(event.percent)
                    step_text.set(f"第 {event.step}/{event.total_steps} 步 · {event.stage}")
                    detail.set(event.message)
                    append_log(line)
                elif kind == "done":
                    probe = payload  # type: ignore[assignment]
                    busy = False
                    _save_engine_choice(args.install_root, probe.variant)
                    progress_value.set(100)
                    status.set(f"{probe.variant.upper()} 环境已通过自检")
                    step_text.set("第 6/6 步 · complete")
                    detail.set(_engine_summary(probe))
                    append_log(record(f"环境就绪：{_engine_summary(probe)}；Python：{probe.python}"))
                    close_button.configure(state="normal")
                    if not args.no_launch:
                        launched_process = launch_desktop_with_engine(
                            probe,
                            app_root,
                            launcher_executable=_launcher_executable(),
                        )
                        root.after(900, root.destroy)
                elif kind == "error":
                    busy = False
                    status.set("环境准备失败")
                    detail.set(f"{payload}。完整诊断已保存在：{log_path}")
                    append_log(record(f"可复制此日志进行诊断：{log_path}"))
                    retry_button.configure(state="normal")
                    choose_button.configure(state="normal")
                    close_button.configure(state="normal")
                    messagebox.showerror(
                        "SPImaging 环境准备失败",
                        f"{payload}\n\n日志：{log_path}",
                        parent=root,
                    )
        except queue.Empty:
            pass
        if root.winfo_exists():
            root.after(100, poll)

    initial = _gui_requested_engine_choice(args)
    append_log(record(f"启动器开始；日志文件：{log_path}"))
    if initial in {"cpu", "cuda"}:
        root.after(150, lambda value=initial: run_variant(value, explicit=True))
    else:
        set_choice_enabled(True)
    root.after(100, poll)
    root.mainloop()
    if launched_process is not None:
        return int(launched_process.wait())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _wait_for_pid(args.wait_for_pid)
    try:
        instance_path = args.install_root.expanduser().resolve() / "metadata" / "launcher-instance.lock"
        with InterProcessLock(instance_path, purpose="启动器单实例"):
            if _legacy_runtime_requested(args):
                return _run_headless(args) if args.headless or args.dry_run else _run_gui(args)
            return _run_engine_headless(args) if args.headless else _run_engine_gui(args)
    except LauncherError as error:
        if os.name == "nt" and not args.headless and not args.dry_run:
            try:
                ctypes.windll.user32.MessageBoxW(
                    None,
                    str(error),
                    "SPImaging 启动失败",
                    0x00000010,
                )
            except (AttributeError, OSError):
                pass
        print(f"SPImaging launcher error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
