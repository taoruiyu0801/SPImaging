"""Trusted cached manifests and rate-limited update-check state."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import os
import uuid
from urllib.parse import urlparse

from .download import DownloadTransport, fetch_bytes
from .errors import LauncherError
from .manifest import ReleaseManifest, compare_semver
from .signing import ManifestTrustPolicy


DEFAULT_CHECK_INTERVAL = timedelta(hours=24)
MANIFEST_SIGNATURE_SUFFIX = ".p7s"


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
    """Persist exact manifest/signature bytes as one atomic cache envelope."""

    CACHE_SCHEMA = 1

    def __init__(self, install_root: Path, trust_policy: ManifestTrustPolicy | None = None) -> None:
        self.metadata_root = install_root.resolve() / "metadata"
        self.manifest_path = self.metadata_root / "release-manifest.json"
        self.active_manifest_path = self.metadata_root / "active-release-manifest.json"
        self.check_path = self.metadata_root / "update-check.json"
        self.trust_policy = trust_policy or ManifestTrustPolicy.current()

    def _encode(self, raw: bytes, signature: bytes | None) -> str:
        value = {
            "cache_schema_version": self.CACHE_SCHEMA,
            "manifest_base64": base64.b64encode(raw).decode("ascii"),
            "signature_base64": None if signature is None else base64.b64encode(signature).decode("ascii"),
        }
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def _decode(self, path: Path) -> tuple[ReleaseManifest, bytes, bytes | None] | None:
        if not path.is_file():
            return None
        try:
            stored = path.read_bytes()
            value = json.loads(stored)
            if isinstance(value, dict) and value.get("cache_schema_version") == self.CACHE_SCHEMA:
                encoded_manifest = value.get("manifest_base64")
                encoded_signature = value.get("signature_base64")
                if not isinstance(encoded_manifest, str):
                    return None
                raw = base64.b64decode(encoded_manifest, validate=True)
                signature = (
                    None
                    if encoded_signature is None
                    else base64.b64decode(encoded_signature, validate=True)
                )
            else:
                # One-time compatibility with the original unsigned-beta
                # cache, which stored the exact manifest directly.
                raw = stored
                signature = None
            manifest = ReleaseManifest.from_json(raw)
            self.trust_policy.verify_manifest(raw, manifest, signature)
            return manifest, raw, signature
        except (OSError, ValueError, TypeError, binascii.Error, json.JSONDecodeError, LauncherError):
            return None

    def load_record(self, *, active: bool = False) -> tuple[ReleaseManifest, bytes, bytes | None] | None:
        return self._decode(self.active_manifest_path if active else self.manifest_path)

    def load(self) -> ReleaseManifest | None:
        record = self.load_record()
        return record[0] if record else None

    def store(self, raw: bytes, manifest: ReleaseManifest, signature: bytes | None = None) -> None:
        # Trust verification occurs before the control file becomes an offline
        # source of truth. The signature is stored in the same atomic envelope.
        self.trust_policy.verify_manifest(raw, manifest, signature)
        _atomic_write(self.manifest_path, self._encode(raw, signature))

    def load_active(self) -> ReleaseManifest | None:
        record = self.load_record(active=True)
        return record[0] if record else None

    def mark_active(
        self,
        raw: bytes,
        manifest: ReleaseManifest,
        signature: bytes | None = None,
    ) -> None:
        self.trust_policy.verify_manifest(raw, manifest, signature)
        _atomic_write(self.active_manifest_path, self._encode(raw, signature))

    def validate_progression(
        self,
        raw: bytes,
        manifest: ReleaseManifest,
        *,
        expected_channel: str | None = None,
    ) -> None:
        """Reject channel/version rollback against cached or active releases."""

        if expected_channel is not None and manifest.channel != expected_channel:
            raise LauncherError(
                f"release manifest channel {manifest.channel} does not match requested {expected_channel} channel"
            )
        candidate = self.load_record()
        active = self.load_record(active=True)
        references = []
        if candidate is not None and candidate[0].channel == manifest.channel:
            references.append(candidate)
        if active is not None:
            references.append(active)
        for reference_manifest, reference_raw, _reference_signature in references:
            if reference_manifest.channel == "stable" and manifest.channel != "stable":
                raise LauncherError("refusing release channel downgrade from stable to beta")
            ordering = compare_semver(manifest.release_version, reference_manifest.release_version)
            if ordering < 0:
                raise LauncherError(
                    f"refusing release-manifest downgrade from {reference_manifest.release_version} "
                    f"to {manifest.release_version}"
                )
            if ordering == 0 and raw != reference_raw:
                raise LauncherError("refusing mutable manifest content at unchanged SemVer precedence")

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
        expected_channel: str | None = None,
    ) -> tuple[ReleaseManifest, bool, str]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise LauncherError("release manifest URL must use HTTPS without embedded credentials")
        cached_record = self.load_record()
        if expected_channel is not None and expected_channel not in {"beta", "stable"}:
            raise LauncherError("expected release channel must be beta or stable")
        if cached_record is not None and expected_channel is not None and cached_record[0].channel != expected_channel:
            cached_record = None
        cached = cached_record[0] if cached_record else None
        if cached is not None and not self.should_check(force=force_check):
            assert cached_record is not None
            self.validate_progression(
                cached_record[1],
                cached,
                expected_channel=expected_channel,
            )
            return cached, False, "24 小时内已检查更新，使用本地清单"
        try:
            raw = fetch_bytes(url, transport)
            signature = (
                fetch_bytes(url + MANIFEST_SIGNATURE_SUFFIX, transport, max_bytes=256 * 1024)
                if self.trust_policy.signature_required
                else None
            )
            manifest = ReleaseManifest.from_json(raw)
            self.validate_progression(raw, manifest, expected_channel=expected_channel)
            # store() authenticates the exact bytes before the atomic write.
            self.store(raw, manifest, signature)
            self.record_check()
            return manifest, True, "已获取并验证最新发布清单"
        except LauncherError as error:
            if cached is None:
                raise
            assert cached_record is not None
            self.validate_progression(
                cached_record[1],
                cached,
                expected_channel=expected_channel,
            )
            # A usable cached release makes an offline failure non-urgent;
            # throttle the next automatic retry to the normal 24-hour window.
            try:
                self.record_check()
            except LauncherError:
                # Metadata pressure must not prevent an otherwise valid
                # offline installation from starting.
                pass
            return cached, False, f"联网检查失败，继续使用已验证的本地清单：{error}"
