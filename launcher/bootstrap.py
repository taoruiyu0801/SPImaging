"""Transactional release provisioning and desktop process construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys
from typing import Callable

from .activation import ActivationManager, HealthCheckRunner
from .archive import safe_extract_zip
from .download import DownloadTransport, ProgressCallback, download_asset
from .errors import HealthCheckError, LauncherError
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


def _release_id(asset: ReleaseAsset) -> str:
    return f"{asset.version}-{asset.variant}"


class Provisioner:
    def __init__(
        self,
        install_root: Path,
        transport: DownloadTransport,
        *,
        cache_root: Path | None = None,
        verifier: SignatureVerifier | None = None,
        health_runner: HealthCheckRunner | None = None,
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
            self._verify_required_paths(root, asset)
            if relocate and asset.relocation is not None:
                self.health_runner.run(root, asset.relocation)
            return self.health_runner.run(root, asset.health_check)

        return check

    def install_asset(self, asset: ReleaseAsset, *, force: bool = False) -> Path:
        release_id = _release_id(asset)
        active = self.manager.active_record(asset.component)
        active_path = self.manager.active_path(asset.component)
        if not force and active_path is not None and active and active.get("active") == release_id:
            self._health(asset)(active_path)
            return active_path
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
            return self.manager.activate(
                asset.component,
                release_id,
                staged,
                health_check=self._health(asset),
                prepare_and_health=self._health(asset, relocate=True),
                metadata={
                    "asset_id": asset.asset_id,
                    "version": asset.version,
                    "variant": asset.variant,
                    "archive_sha256": asset.archive_sha256,
                },
                replace_existing=force,
            )
        except Exception:
            if staged.exists():
                self.manager.discard_staging(staged)
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
        try:
            runtime_path = self.install_asset(selection.asset, force=force)
        except HealthCheckError as error:
            if selection.selected_variant != "cuda":
                raise
            cpu = manifest.asset("runtime", "cpu")
            runtime_path = self.install_asset(cpu, force=force)
            selection = RuntimeSelection(
                asset=cpu,
                requested_variant=selection.requested_variant,
                selected_variant="cpu",
                fallback=True,
                reason=f"CUDA 自检失败（{error}）；已回退 CPU",
            )
        app_path = self.install_asset(manifest.asset("app", "universal"), force=force)
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
    environment = os.environ.copy()
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(result.app_path) + (os.pathsep + prior if prior else "")
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
