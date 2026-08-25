"""Transactional release provisioning and desktop process construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Callable

from .activation import ActivationManager, ActivationRequest, HealthCheckRunner
from .archive import safe_extract_zip
from .download import DownloadTransport, ProgressCallback, download_asset
from .errors import DownloadError, HealthCheckError, LauncherError
from .manifest import (
    NvidiaCapability,
    ReleaseAsset,
    ReleaseManifest,
    RuntimeSelection,
    select_runtime_asset,
)
from .signing import SignatureVerifier


@dataclass(frozen=True)
class ProvisionResult:
    release_version: str
    runtime_path: Path
    app_path: Path
    runtime_variant: str
    fallback: bool
    reason: str


@dataclass(frozen=True)
class _PreparedAsset:
    asset: ReleaseAsset
    release_id: str
    path: Path
    staged: Path | None
    replace_existing: bool = False


class _CudaRuntimeHealthError(HealthCheckError):
    """Distinguish CUDA self-check failures from application smoke failures."""


def _release_id(asset: ReleaseAsset) -> str:
    return f"{asset.version}-{asset.variant}"


def _private_python_environment(runtime_root: Path, app_root: Path) -> dict[str, str]:
    """Build an environment that cannot inherit an activated user Python."""

    environment = os.environ.copy()
    for key in tuple(environment):
        upper = key.upper()
        if upper.startswith("CONDA_") or upper in {
            "PYTHONHOME",
            "PYTHONINSPECT",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "QML2_IMPORT_PATH",
            "QT_PLUGIN_PATH",
            "VIRTUAL_ENV",
            "_CE_CONDA",
            "_CE_M",
        }:
            environment.pop(key, None)
    environment["PYTHONPATH"] = str(app_root)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    private_bins = (
        runtime_root,
        runtime_root / "Scripts",
        runtime_root / "Library" / "bin",
    )
    prior_path = environment.get("PATH", "")
    environment["PATH"] = os.pathsep.join(
        [*(str(path) for path in private_bins if path.is_dir()), prior_path]
    ).rstrip(os.pathsep)
    return environment


class Provisioner:
    def __init__(
        self,
        install_root: Path,
        transport: DownloadTransport,
        *,
        cache_root: Path | None = None,
        verifier: SignatureVerifier | None = None,
        health_runner: HealthCheckRunner | None = None,
        desktop_smoke: Callable[[Path, Path, ReleaseManifest], object] | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.manager = ActivationManager(install_root)
        self.cache_root = (
            Path(cache_root).expanduser().resolve()
            if cache_root is not None
            else self.manager.install_root / "cache"
        )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.transport = transport
        self.verifier = verifier
        self.health_runner = health_runner or HealthCheckRunner()
        self.desktop_smoke = desktop_smoke or self._run_desktop_smoke
        self.progress = progress

    def _verify_required_paths(self, root: Path, asset: ReleaseAsset) -> None:
        for relative in asset.required_paths:
            candidate = root.joinpath(*relative.split("/"))
            try:
                candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
            except (OSError, ValueError) as error:
                raise HealthCheckError(f"required release path is missing or unsafe: {relative}") from error

    def _health(self, asset: ReleaseAsset, *, relocate: bool = False) -> Callable[[Path], object]:
        def check(root: Path) -> object:
            try:
                self._verify_required_paths(root, asset)
                if relocate and asset.relocation is not None:
                    self.health_runner.run(root, asset.relocation)
                return self.health_runner.run(root, asset.health_check)
            except HealthCheckError as error:
                if asset.component == "runtime" and asset.variant == "cuda":
                    raise _CudaRuntimeHealthError(str(error)) from error
                raise

        return check

    def _run_desktop_smoke(self, runtime_root: Path, app_root: Path, manifest: ReleaseManifest) -> object:
        """Construct the packaged PySide6 window with the selected runtime."""

        executable = runtime_root.joinpath(*manifest.launch.console_executable.split("/"))
        try:
            executable.resolve(strict=True).relative_to(runtime_root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise HealthCheckError("selected runtime is missing its console Python for the app smoke test") from error
        environment = _private_python_environment(runtime_root, app_root)
        environment["QT_QPA_PLATFORM"] = "offscreen"
        command = [str(executable), "-m", manifest.launch.app_module, "--smoke-test"]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            completed = subprocess.run(
                command,
                cwd=app_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=120,
                creationflags=creationflags,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise HealthCheckError(f"desktop smoke test could not run: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
            raise HealthCheckError(f"desktop smoke test failed with exit code {completed.returncode}: {detail}")
        return completed

    def _needs_install(self, asset: ReleaseAsset, *, force: bool) -> bool:
        record = self.manager.active_record(asset.component) or {}
        path = self.manager.active_path(asset.component)
        return force or path is None or record.get("active") != _release_id(asset)

    @staticmethod
    def _verify_active_identity(record: dict[str, object], asset: ReleaseAsset) -> None:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return
        installed_digest = metadata.get("archive_sha256")
        if isinstance(installed_digest, str) and installed_digest != asset.archive_sha256:
            raise LauncherError(
                f"component version {asset.version} was republished with a different archive hash; "
                "increment the component version instead"
            )

    @staticmethod
    def _ensure_volume_space(directory: Path, required: int, label: str) -> int:
        directory.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(directory).free
        margin = max(64 * 1024 * 1024, required // 20)
        if free < required + margin:
            raise DownloadError(
                f"insufficient {label} disk space: need {required + margin} bytes, have {free}"
            )
        return free

    @staticmethod
    def _same_volume(left: Path, right: Path) -> bool:
        try:
            return os.stat(left).st_dev == os.stat(right).st_dev
        except OSError:
            left_drive = os.path.splitdrive(str(left.resolve()))[0].casefold()
            right_drive = os.path.splitdrive(str(right.resolve()))[0].casefold()
            return bool(left_drive) and left_drive == right_drive

    def _preflight_assets(self, assets: tuple[ReleaseAsset, ...], *, force: bool) -> None:
        pending = tuple(asset for asset in assets if self._needs_install(asset, force=force))
        if not pending:
            return
        # Cache and install roots may be on different volumes. Check both
        # before the first network byte is requested.
        cache_required = sum(
            asset.archive_size * 2 + (asset.signature.size or 0)
            for asset in pending
        )
        unpacked_required = sum(asset.unpacked_size for asset in pending)
        # A first install has no previous component to roll back to, so reserve
        # one additional unpacked slot. During an update the active directory
        # already occupies and supplies that rollback slot.
        rollback_reserve = sum(
            asset.unpacked_size
            for asset in pending
            if self.manager.active_path(asset.component) is None
        )
        cache_free = self._ensure_volume_space(self.cache_root, cache_required, "download cache")
        install_required = unpacked_required + rollback_reserve
        install_free = self._ensure_volume_space(
            self.manager.staging_root,
            install_required,
            "installation/staging and rollback",
        )
        if self._same_volume(self.cache_root, self.manager.staging_root):
            combined = cache_required + install_required
            margin = max(64 * 1024 * 1024, combined // 20)
            free = min(cache_free, install_free)
            if free < combined + margin:
                raise DownloadError(
                    f"insufficient shared cache/install disk space: need {combined + margin} bytes, have {free}"
                )

    def _prepare_asset(self, asset: ReleaseAsset, *, force: bool = False) -> _PreparedAsset:
        release_id = _release_id(asset)
        active = self.manager.active_record(asset.component)
        active_path = self.manager.active_path(asset.component)
        if active_path is not None and active and active.get("active") == release_id:
            self._verify_active_identity(active, asset)
            if not force:
                self._health(asset)(active_path)
                return _PreparedAsset(asset, release_id, active_path, None)
        archive = download_asset(
            asset,
            self.cache_root,
            self.transport,
            verifier=self.verifier,
            progress=self.progress,
        )
        staged = self.manager.create_staging(asset.component)
        try:
            safe_extract_zip(archive, staged, max_unpacked_size=asset.unpacked_size)
            self._verify_required_paths(staged, asset)
        except Exception:
            if staged.exists():
                self.manager.discard_staging(staged)
            raise
        target = self.manager.release_path(asset.component, release_id)
        return _PreparedAsset(asset, release_id, target, staged, force or target.exists())

    def _request(self, prepared: _PreparedAsset) -> ActivationRequest:
        if prepared.staged is None:
            raise LauncherError("internal error: active asset cannot be staged for activation")
        asset = prepared.asset
        return ActivationRequest(
            asset.component,
            prepared.release_id,
            prepared.staged,
            self._health(asset),
            self._health(asset, relocate=True),
            {
                "asset_id": asset.asset_id,
                "version": asset.version,
                "variant": asset.variant,
                "archive_sha256": asset.archive_sha256,
                "unpacked_size": asset.unpacked_size,
            },
            prepared.replace_existing,
        )

    def install_asset(self, asset: ReleaseAsset, *, force: bool = False) -> Path:
        """Install one component; full releases should use :meth:`provision`."""

        self._preflight_assets((asset,), force=force)
        prepared = self._prepare_asset(asset, force=force)
        if prepared.staged is None:
            return prepared.path
        try:
            return self.manager.activate_many([self._request(prepared)])[asset.component]
        except Exception:
            if prepared.staged.exists():
                self.manager.discard_staging(prepared.staged)
            raise

    def _provision_pair(
        self,
        manifest: ReleaseManifest,
        runtime_asset: ReleaseAsset,
        *,
        force: bool,
    ) -> tuple[Path, Path]:
        app_asset = manifest.asset("app", "universal")
        self._preflight_assets((runtime_asset, app_asset), force=force)
        prepared: list[_PreparedAsset] = []
        try:
            runtime = self._prepare_asset(runtime_asset, force=force)
            prepared.append(runtime)
            app = self._prepare_asset(app_asset, force=force)
            prepared.append(app)
            requests = [self._request(item) for item in prepared if item.staged is not None]

            def combined(changed: dict[str, Path]) -> object:
                runtime_path = changed.get("runtime", runtime.path)
                app_path = changed.get("app", app.path)
                return self.desktop_smoke(runtime_path, app_path, manifest)

            if requests:
                changed = self.manager.activate_many(requests, final_health_check=combined)
                return changed.get("runtime", runtime.path), changed.get("app", app.path)
            combined({})
            return runtime.path, app.path
        except Exception:
            for item in prepared:
                if item.staged is not None and item.staged.exists():
                    self.manager.discard_staging(item.staged)
            raise

    def provision(
        self,
        manifest: ReleaseManifest,
        preference: str,
        capability: NvidiaCapability,
        *,
        force: bool = False,
    ) -> ProvisionResult:
        selection = select_runtime_asset(manifest, preference, capability)
        snapshot = self.manager.snapshot_state()
        try:
            try:
                runtime_path, app_path = self._provision_pair(manifest, selection.asset, force=force)
            except _CudaRuntimeHealthError as error:
                if selection.selected_variant != "cuda":
                    raise
                self.manager.restore_state(snapshot)
                cpu = manifest.asset("runtime", "cpu")
                runtime_path, app_path = self._provision_pair(manifest, cpu, force=force)
                selection = RuntimeSelection(
                    asset=cpu,
                    requested_variant=selection.requested_variant,
                    selected_variant="cpu",
                    fallback=True,
                    reason=f"CUDA 自检失败（{error}）；已回退 CPU",
                )
        except Exception:
            self.manager.restore_state(snapshot)
            raise
        return ProvisionResult(
            manifest.release_version,
            runtime_path,
            app_path,
            selection.selected_variant,
            selection.fallback,
            selection.reason,
        )


def build_desktop_command(manifest: ReleaseManifest, result: ProvisionResult) -> tuple[list[str], dict[str, str]]:
    executable = result.runtime_path.joinpath(*manifest.launch.runtime_executable.split("/"))
    if not executable.is_file():
        # A console interpreter is a useful source-run fallback, but remains
        # inside the private runtime rather than using a system Python.
        executable = result.runtime_path.joinpath(*manifest.launch.console_executable.split("/"))
    if not executable.is_file():
        raise LauncherError("private runtime does not contain its declared Python executable")
    command = [str(executable), "-m", manifest.launch.app_module, *manifest.launch.arguments]
    environment = _private_python_environment(result.runtime_path, result.app_path)
    environment["SPIMAGING_INSTALL_ROOT"] = str(result.runtime_path.parent.parent.parent.parent)
    environment["SPIMAGING_RUNTIME_VARIANT"] = result.runtime_variant
    return command, environment


def launch_desktop(manifest: ReleaseManifest, result: ProvisionResult) -> subprocess.Popen[bytes]:
    command, environment = build_desktop_command(manifest, result)
    if getattr(sys, "frozen", False):
        environment["SPIMAGING_LAUNCHER_EXE"] = str(Path(sys.executable).resolve())
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    return subprocess.Popen(
        command,
        cwd=result.app_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
