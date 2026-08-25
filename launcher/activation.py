"""Health-checked staging, atomic activation state, and one-step rollback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path
import os
import re
import shutil
import subprocess
import uuid
from typing import Any, Callable

from .errors import ActivationError, HealthCheckError
from .manifest import HealthCheck


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


@dataclass(frozen=True)
class HealthResult:
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ActivationSnapshot:
    """In-memory copy of the authoritative activation state."""

    state_file_existed: bool
    state: dict[str, Any]


@dataclass(frozen=True)
class ActivationRequest:
    """One staged component participating in a multi-component switch."""

    component: str
    release_id: str
    staged: Path
    health_check: Callable[[Path], object]
    prepare_and_health: Callable[[Path], object] | None = None
    metadata: dict[str, Any] | None = None
    replace_existing: bool = False


class HealthCheckRunner:
    def run(self, root: Path, check: HealthCheck | None) -> HealthResult:
        if check is None:
            return HealthResult((), 0, "", "")
        root = root.resolve(strict=True)
        executable = root.joinpath(*check.executable.split("/"))
        try:
            executable.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise HealthCheckError(f"health-check executable is missing or unsafe: {check.executable}") from error
        command = (str(executable), *check.arguments)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=check.timeout_seconds,
                creationflags=creationflags,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise HealthCheckError(f"health check could not run: {error}") from error
        result = HealthResult(command, completed.returncode, completed.stdout, completed.stderr)
        if completed.returncode != check.expected_exit_code:
            detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
            raise HealthCheckError(
                f"health check returned {completed.returncode}, expected {check.expected_exit_code}: {detail}"
            )
        return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ActivationManager:
    """Own release directories and switch a small state file atomically."""

    STATE_SCHEMA = 1

    def __init__(self, install_root: Path) -> None:
        self.install_root = install_root.resolve()
        self.staging_root = self.install_root / "staging"
        self.components_root = self.install_root / "components"
        self.state_path = self.install_root / "state.json"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.components_root.mkdir(parents=True, exist_ok=True)

    def _validate_id(self, value: str, field: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ActivationError(f"unsafe {field}: {value!r}")
        return value

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": self.STATE_SCHEMA, "components": {}}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ActivationError(f"activation state is unreadable: {error}") from error
        if state.get("schema_version") != self.STATE_SCHEMA or not isinstance(state.get("components"), dict):
            raise ActivationError("activation state schema is invalid")
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        self.install_root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ActivationError(f"could not atomically write activation state: {error}") from error

    def create_staging(self, component: str) -> Path:
        component = self._validate_id(component, "component")
        path = self.staging_root / f"{component}-{uuid.uuid4().hex}"
        path.mkdir(parents=False, exist_ok=False)
        return path

    def release_path(self, component: str, release_id: str) -> Path:
        component = self._validate_id(component, "component")
        release_id = self._validate_id(release_id, "release_id")
        return self.components_root / component / "releases" / release_id

    def active_record(self, component: str) -> dict[str, Any] | None:
        component = self._validate_id(component, "component")
        record = self._read_state()["components"].get(component)
        return record if isinstance(record, dict) else None

    def active_path(self, component: str) -> Path | None:
        record = self.active_record(component)
        if not record or not isinstance(record.get("active"), str):
            return None
        path = self.release_path(component, record["active"])
        return path if path.is_dir() else None

    def snapshot_state(self) -> ActivationSnapshot:
        """Capture state so a wider install transaction can restore it."""

        return ActivationSnapshot(self.state_path.is_file(), deepcopy(self._read_state()))

    def restore_state(self, snapshot: ActivationSnapshot) -> None:
        """Restore a previously captured state using the same atomic writer."""

        if not isinstance(snapshot, ActivationSnapshot):
            raise ActivationError("activation snapshot has an invalid type")
        state = deepcopy(snapshot.state)
        if state.get("schema_version") != self.STATE_SCHEMA or not isinstance(state.get("components"), dict):
            raise ActivationError("activation snapshot schema is invalid")
        if snapshot.state_file_existed:
            self._write_state(state)
            return
        try:
            self.state_path.unlink(missing_ok=True)
        except OSError as error:
            raise ActivationError(f"could not restore empty activation state: {error}") from error

    def activate_many(
        self,
        requests: list[ActivationRequest],
        *,
        final_health_check: Callable[[dict[str, Path]], object] | None = None,
    ) -> dict[str, Path]:
        """Place and health-check components, then publish one state update.

        Directories may need to be moved to their final paths before conda's
        relocation check can run.  Until every component and the combined
        desktop smoke test pass, the authoritative state file is untouched.
        Any failure restores same-version repairs from quarantine.
        """

        if not requests:
            paths: dict[str, Path] = {}
            if final_health_check is not None:
                final_health_check(paths)
            return paths
        components = [self._validate_id(item.component, "component") for item in requests]
        if len(set(components)) != len(components):
            raise ActivationError("activation transaction contains duplicate components")
        snapshot = self.snapshot_state()
        placements: list[tuple[ActivationRequest, Path, Path | None, bool]] = []
        result: dict[str, Path] = {}
        try:
            for request in requests:
                component = self._validate_id(request.component, "component")
                release_id = self._validate_id(request.release_id, "release_id")
                staged = request.staged.resolve(strict=True)
                try:
                    staged.relative_to(self.staging_root)
                except ValueError as error:
                    raise ActivationError("staged directory is outside the managed staging root") from error
                target = self.release_path(component, release_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                quarantine: Path | None = None
                created_target = False
                staged_check = request.prepare_and_health or request.health_check
                if target.exists() and not request.replace_existing:
                    request.health_check(target)
                    shutil.rmtree(staged)
                else:
                    if target.exists():
                        quarantine_root = target.parent.parent / "quarantine"
                        quarantine_root.mkdir(parents=True, exist_ok=True)
                        quarantine = quarantine_root / f"{release_id}-{uuid.uuid4().hex}"
                        os.replace(target, quarantine)
                    try:
                        os.replace(staged, target)
                        created_target = True
                        staged_check(target)
                    except Exception:
                        if target.exists():
                            shutil.rmtree(target)
                        if quarantine is not None and quarantine.exists():
                            os.replace(quarantine, target)
                        raise
                placements.append((request, target, quarantine, created_target))
                result[component] = target

            if final_health_check is not None:
                final_health_check(result)

            state = deepcopy(snapshot.state)
            for request, _target, _quarantine, _created in placements:
                old = state["components"].get(request.component, {})
                old_active = old.get("active") if isinstance(old, dict) else None
                state["components"][request.component] = {
                    "active": request.release_id,
                    "previous": old_active if old_active != request.release_id else old.get("previous"),
                    "activated_at": _utc_now(),
                    "metadata": request.metadata or {},
                }
            self._write_state(state)
        except Exception as error:
            for request, target, quarantine, created_target in reversed(placements):
                try:
                    if created_target and target.exists():
                        shutil.rmtree(target)
                    if quarantine is not None and quarantine.exists():
                        os.replace(quarantine, target)
                except OSError as rollback_error:
                    raise ActivationError(
                        f"activation failed ({error}) and filesystem rollback failed: {rollback_error}"
                    ) from rollback_error
            self.restore_state(snapshot)
            if isinstance(error, (ActivationError, HealthCheckError)):
                raise
            raise ActivationError(f"multi-component activation failed: {error}") from error

        for _request, _target, quarantine, _created in placements:
            if quarantine is not None and quarantine.exists():
                shutil.rmtree(quarantine, ignore_errors=True)
        return result

    def activate(
        self,
        component: str,
        release_id: str,
        staged: Path,
        *,
        health_check: Callable[[Path], object],
        prepare_and_health: Callable[[Path], object] | None = None,
        metadata: dict[str, Any] | None = None,
        replace_existing: bool = False,
    ) -> Path:
        request = ActivationRequest(
            component,
            release_id,
            staged,
            health_check,
            prepare_and_health,
            metadata,
            replace_existing,
        )
        return self.activate_many([request])[component]

    def rollback(self, component: str, *, health_check: Callable[[Path], object]) -> Path:
        component = self._validate_id(component, "component")
        state = self._read_state()
        record = state["components"].get(component)
        if not isinstance(record, dict) or not isinstance(record.get("previous"), str):
            raise ActivationError(f"no previous {component} release is available")
        previous = record["previous"]
        target = self.release_path(component, previous)
        if not target.is_dir():
            raise ActivationError(f"previous {component} release directory is missing")
        health_check(target)
        current = record.get("active")
        record.update({"active": previous, "previous": current, "activated_at": _utc_now()})
        self._write_state(state)
        return target

    def discard_staging(self, staged: Path) -> None:
        """Remove only a verified child of this manager's staging directory."""

        try:
            resolved = staged.resolve()
            resolved.relative_to(self.staging_root)
        except (OSError, ValueError) as error:
            raise ActivationError("refusing to remove unmanaged staging path") from error
        if resolved.exists():
            shutil.rmtree(resolved)
