"""Cached manifest and rate-limited update-check state."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import os
import uuid
from urllib.parse import urlparse

from .download import DownloadTransport, fetch_bytes
from .errors import LauncherError
from .manifest import ReleaseManifest


DEFAULT_CHECK_INTERVAL = timedelta(hours=24)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise LauncherError(f"could not persist launcher metadata: {error}") from error


class ManifestCache:
    def __init__(self, install_root: Path) -> None:
        self.metadata_root = install_root.resolve() / "metadata"
        self.manifest_path = self.metadata_root / "release-manifest.json"
        self.active_manifest_path = self.metadata_root / "active-release-manifest.json"
        self.check_path = self.metadata_root / "update-check.json"

    def load(self) -> ReleaseManifest | None:
        if not self.manifest_path.is_file():
            return None
        try:
            return ReleaseManifest.from_json(self.manifest_path.read_bytes())
        except (OSError, LauncherError):
            return None

    def store(self, raw: bytes, manifest: ReleaseManifest) -> None:
        # Re-parse the exact bytes before making them the offline source of truth.
        if ReleaseManifest.from_json(raw) != manifest:
            raise LauncherError("refusing to cache a manifest that changed after validation")
        _atomic_write(self.manifest_path, raw.decode("utf-8"))

    def load_active(self) -> ReleaseManifest | None:
        if not self.active_manifest_path.is_file():
            return None
        try:
            return ReleaseManifest.from_json(self.active_manifest_path.read_bytes())
        except (OSError, LauncherError):
            return None

    def mark_active(self, raw: bytes, manifest: ReleaseManifest) -> None:
        if ReleaseManifest.from_json(raw) != manifest:
            raise LauncherError("refusing to activate mismatched manifest bytes")
        _atomic_write(self.active_manifest_path, raw.decode("utf-8"))

    def last_checked(self) -> datetime | None:
        try:
            value = json.loads(self.check_path.read_text(encoding="utf-8"))["last_checked_at"]
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def should_check(self, *, now: datetime | None = None, force: bool = False) -> bool:
        if force:
            return True
        last = self.last_checked()
        return last is None or (now or _now()) - last >= DEFAULT_CHECK_INTERVAL

    def record_check(self, *, now: datetime | None = None) -> None:
        checked = (now or _now()).isoformat().replace("+00:00", "Z")
        _atomic_write(self.check_path, json.dumps({"schema_version": 1, "last_checked_at": checked}, indent=2) + "\n")

    def resolve(
        self,
        url: str,
        transport: DownloadTransport,
        *,
        force_check: bool = False,
    ) -> tuple[ReleaseManifest, bool, str]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise LauncherError("release manifest URL must use HTTPS without embedded credentials")
        cached = self.load()
        if cached is not None and not self.should_check(force=force_check):
            return cached, False, "24 小时内已检查更新，使用本地清单"
        try:
            raw = fetch_bytes(url, transport)
            manifest = ReleaseManifest.from_json(raw)
            self.store(raw, manifest)
            self.record_check()
            return manifest, True, "已获取并验证最新发布清单"
        except LauncherError as error:
            if cached is None:
                raise
            return cached, False, f"联网检查失败，继续使用已验证的本地清单：{error}"
