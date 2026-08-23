"""Versioned release-manifest contract and CPU/CUDA selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .errors import ManifestError


SCHEMA_VERSION = 1
PLATFORM = "windows-x86_64"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$" )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field} must be an object")
    return value


def _text(value: Any, field: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    result = value.strip()
    if pattern is not None and pattern.fullmatch(result) is None:
        raise ManifestError(f"{field} has an invalid value")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{field} must be a boolean")
    return value


def _safe_relative(value: Any, field: str) -> str:
    result = _text(value, field)
    if "\\" in result or ":" in result or "//" in result:
        raise ManifestError(f"{field} must use a safe relative POSIX path")
    path = PurePosixPath(result)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ManifestError(f"{field} must use a safe relative POSIX path")
    return path.as_posix()


def _https_url(value: Any, field: str) -> str:
    result = _text(value, field)
    parsed = urlparse(result)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ManifestError(f"{field} must be an HTTPS URL without embedded credentials")
    return result


def _sha256(value: Any, field: str) -> str:
    result = _text(value, field).lower()
    if _SHA256_RE.fullmatch(result) is None:
        raise ManifestError(f"{field} must be a lowercase SHA-256 digest")
    return result


@dataclass(frozen=True)
class AssetPart:
    name: str
    url: str
    size: int
    sha256: str

    @classmethod
    def from_dict(cls, value: Any, field: str) -> "AssetPart":
        data = _mapping(value, field)
        name = _safe_relative(data.get("name"), f"{field}.name")
        if "/" in name:
            raise ManifestError(f"{field}.name must be a file name, not a path")
        return cls(
            name=name,
            url=_https_url(data.get("url"), f"{field}.url"),
            size=_integer(data.get("size"), f"{field}.size", minimum=1),
            sha256=_sha256(data.get("sha256"), f"{field}.sha256"),
        )


@dataclass(frozen=True)
class SignatureRequirement:
    kind: str
    required: bool
    signer_thumbprint: str | None = None
    file_name: str | None = None
    url: str | None = None
    size: int | None = None
    sha256: str | None = None

    @classmethod
    def from_dict(cls, value: Any, field: str) -> "SignatureRequirement":
        data = _mapping(value, field)
        kind = _text(data.get("kind"), f"{field}.kind").lower()
        if kind not in {"none", "authenticode", "cms-detached"}:
            raise ManifestError(f"{field}.kind is unsupported")
        required = _boolean(data.get("required"), f"{field}.required")
        if kind == "none":
            if required:
                raise ManifestError(f"{field} cannot require a 'none' signature")
            return cls(kind=kind, required=False)
        if not required:
            raise ManifestError(f"{field} must require any declared publisher signature")
        thumbprint = _text(data.get("signer_thumbprint"), f"{field}.signer_thumbprint")
        normalized_thumbprint = re.sub(r"\s+", "", thumbprint).upper()
        if re.fullmatch(r"[0-9A-F]{40}|[0-9A-F]{64}", normalized_thumbprint) is None:
            raise ManifestError(f"{field}.signer_thumbprint must be a SHA-1 or SHA-256 certificate thumbprint")
        if kind == "authenticode":
            return cls(kind=kind, required=required, signer_thumbprint=normalized_thumbprint)
        file_name = _safe_relative(data.get("file_name"), f"{field}.file_name")
        if "/" in file_name:
            raise ManifestError(f"{field}.file_name must be a file name")
        return cls(
            kind=kind,
            required=required,
            signer_thumbprint=normalized_thumbprint,
            file_name=file_name,
            url=_https_url(data.get("url"), f"{field}.url"),
            size=_integer(data.get("size"), f"{field}.size", minimum=1),
            sha256=_sha256(data.get("sha256"), f"{field}.sha256"),
        )


@dataclass(frozen=True)
class HealthCheck:
    executable: str
    arguments: tuple[str, ...]
    expected_exit_code: int = 0
    timeout_seconds: int = 60

    @classmethod
    def from_dict(cls, value: Any, field: str) -> "HealthCheck":
        data = _mapping(value, field)
        arguments = data.get("arguments", [])
        if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
            raise ManifestError(f"{field}.arguments must be an array of strings")
        return cls(
            executable=_safe_relative(data.get("executable"), f"{field}.executable"),
            arguments=tuple(arguments),
            expected_exit_code=_integer(data.get("expected_exit_code", 0), f"{field}.expected_exit_code", minimum=0),
            timeout_seconds=_integer(data.get("timeout_seconds", 60), f"{field}.timeout_seconds", minimum=1),
        )


@dataclass(frozen=True)
class ReleaseAsset:
    asset_id: str
    component: str
    variant: str
    platform: str
    version: str
    archive_name: str
    archive_format: str
    archive_size: int
    archive_sha256: str
    unpacked_size: int
    parts: tuple[AssetPart, ...]
    required_paths: tuple[str, ...]
    relocation: HealthCheck | None
    health_check: HealthCheck | None
    signature: SignatureRequirement
    min_nvidia_driver: str | None = None

    @classmethod
    def from_dict(cls, value: Any, field: str) -> "ReleaseAsset":
        data = _mapping(value, field)
        component = _text(data.get("component"), f"{field}.component").lower()
        if component not in {"runtime", "app"}:
            raise ManifestError(f"{field}.component must be runtime or app")
        variant = _text(data.get("variant"), f"{field}.variant").lower()
        allowed_variants = {"cpu", "cuda"} if component == "runtime" else {"universal"}
        if variant not in allowed_variants:
            raise ManifestError(f"{field}.variant is invalid for {component}")
        archive_format = _text(data.get("archive_format"), f"{field}.archive_format").lower()
        if archive_format != "zip":
            raise ManifestError(f"{field}.archive_format must be zip")
        archive_name = _safe_relative(data.get("archive_name"), f"{field}.archive_name")
        if "/" in archive_name:
            raise ManifestError(f"{field}.archive_name must be a file name")
        parts_value = data.get("parts")
        if not isinstance(parts_value, list) or not parts_value:
            raise ManifestError(f"{field}.parts must contain at least one part")
        parts = tuple(AssetPart.from_dict(part, f"{field}.parts[{index}]") for index, part in enumerate(parts_value))
        if len({part.name.casefold() for part in parts}) != len(parts):
            raise ManifestError(f"{field}.parts contains duplicate file names")
        archive_size = _integer(data.get("archive_size"), f"{field}.archive_size", minimum=1)
        if sum(part.size for part in parts) != archive_size:
            raise ManifestError(f"{field}.archive_size must equal the sum of part sizes")
        required_paths_value = data.get("required_paths", [])
        if not isinstance(required_paths_value, list) or not required_paths_value:
            raise ManifestError(f"{field}.required_paths must contain at least one path")
        required_paths = tuple(
            _safe_relative(item, f"{field}.required_paths[{index}]")
            for index, item in enumerate(required_paths_value)
        )
        min_driver = data.get("min_nvidia_driver")
        if min_driver is not None:
            min_driver = _text(min_driver, f"{field}.min_nvidia_driver")
            if re.fullmatch(r"\d+(?:\.\d+){0,3}", min_driver) is None:
                raise ManifestError(f"{field}.min_nvidia_driver is invalid")
        if variant != "cuda" and min_driver is not None:
            raise ManifestError(f"{field}.min_nvidia_driver is only valid for CUDA assets")
        health_value = data.get("health_check")
        health = None if health_value is None else HealthCheck.from_dict(health_value, f"{field}.health_check")
        relocation_value = data.get("relocation")
        relocation = (
            None
            if relocation_value is None
            else HealthCheck.from_dict(relocation_value, f"{field}.relocation")
        )
        if component != "runtime" and relocation is not None:
            raise ManifestError(f"{field}.relocation is only valid for runtime assets")
        return cls(
            asset_id=_text(data.get("asset_id"), f"{field}.asset_id", pattern=_ID_RE),
            component=component,
            variant=variant,
            platform=_text(data.get("platform"), f"{field}.platform"),
            version=_text(data.get("version"), f"{field}.version", pattern=_VERSION_RE),
            archive_name=archive_name,
            archive_format=archive_format,
            archive_size=archive_size,
            archive_sha256=_sha256(data.get("archive_sha256"), f"{field}.archive_sha256"),
            unpacked_size=_integer(data.get("unpacked_size"), f"{field}.unpacked_size", minimum=1),
            parts=parts,
            required_paths=required_paths,
            relocation=relocation,
            health_check=health,
            signature=SignatureRequirement.from_dict(data.get("signature"), f"{field}.signature"),
            min_nvidia_driver=min_driver,
        )


@dataclass(frozen=True)
class LaunchSpec:
    runtime_executable: str
    console_executable: str
    app_module: str
    arguments: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "LaunchSpec":
        data = _mapping(value, "launch")
        arguments = data.get("arguments", [])
        if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
            raise ManifestError("launch.arguments must be an array of strings")
        app_module = _text(data.get("app_module"), "launch.app_module")
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", app_module) is None:
            raise ManifestError("launch.app_module must be a Python module name")
        return cls(
            runtime_executable=_safe_relative(data.get("runtime_executable"), "launch.runtime_executable"),
            console_executable=_safe_relative(data.get("console_executable", "python.exe"), "launch.console_executable"),
            app_module=app_module,
            arguments=tuple(arguments),
        )


@dataclass(frozen=True)
class ReleaseManifest:
    schema_version: int
    product: str
    release_version: str
    channel: str
    published_at: str
    unsigned_beta: bool
    assets: tuple[ReleaseAsset, ...]
    launch: LaunchSpec

    @classmethod
    def from_dict(cls, value: Any) -> "ReleaseManifest":
        data = _mapping(value, "manifest")
        schema_version = _integer(data.get("schema_version"), "schema_version", minimum=1)
        if schema_version != SCHEMA_VERSION:
            raise ManifestError(f"unsupported release manifest schema_version {schema_version}")
        release_version = _text(data.get("release_version"), "release_version", pattern=_VERSION_RE)
        channel = _text(data.get("channel"), "channel").lower()
        if channel not in {"beta", "stable"}:
            raise ManifestError("channel must be beta or stable")
        unsigned_beta = _boolean(data.get("unsigned_beta"), "unsigned_beta")
        if unsigned_beta and channel != "beta":
            raise ManifestError("only beta releases may be marked unsigned_beta")
        published_at = _text(data.get("published_at"), "published_at")
        try:
            datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ManifestError("published_at must be an ISO-8601 timestamp") from error
        assets_value = data.get("assets")
        if not isinstance(assets_value, list) or not assets_value:
            raise ManifestError("assets must contain release assets")
        assets = tuple(ReleaseAsset.from_dict(asset, f"assets[{index}]") for index, asset in enumerate(assets_value))
        if len({asset.asset_id.casefold() for asset in assets}) != len(assets):
            raise ManifestError("asset_id values must be unique")
        for asset in assets:
            if asset.version != release_version:
                raise ManifestError(f"asset {asset.asset_id} does not match release_version")
            if asset.platform != PLATFORM:
                raise ManifestError(f"asset {asset.asset_id} targets unsupported platform {asset.platform}")
            if not unsigned_beta and not asset.signature.required:
                raise ManifestError("signed releases must require a signature for every asset")
        components = {(asset.component, asset.variant) for asset in assets}
        if ("runtime", "cpu") not in components or ("app", "universal") not in components:
            raise ManifestError("manifest requires CPU runtime and universal app assets")
        return cls(
            schema_version=schema_version,
            product=_text(data.get("product"), "product"),
            release_version=release_version,
            channel=channel,
            published_at=published_at,
            unsigned_beta=unsigned_beta,
            assets=assets,
            launch=LaunchSpec.from_dict(data.get("launch")),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "ReleaseManifest":
        try:
            data = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ManifestError("release manifest is not valid UTF-8 JSON") from error
        return cls.from_dict(data)

    def asset(self, component: str, variant: str) -> ReleaseAsset:
        matches = [item for item in self.assets if item.component == component and item.variant == variant]
        if len(matches) != 1:
            raise ManifestError(f"manifest must contain exactly one {component}/{variant} asset")
        return matches[0]


@dataclass(frozen=True)
class NvidiaCapability:
    available: bool
    driver_version: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class RuntimeSelection:
    asset: ReleaseAsset
    requested_variant: str
    selected_variant: str
    fallback: bool
    reason: str


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in value.split("."))
    return parts + (0,) * (4 - len(parts))


def select_runtime_asset(
    manifest: ReleaseManifest,
    preference: str,
    capability: NvidiaCapability,
) -> RuntimeSelection:
    """Choose CUDA when usable and otherwise return CPU with an explicit reason."""

    requested = preference.strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ManifestError("runtime preference must be auto, cuda, or cpu")
    cpu = manifest.asset("runtime", "cpu")
    cuda_assets = [item for item in manifest.assets if item.component == "runtime" and item.variant == "cuda"]
    cuda = cuda_assets[0] if len(cuda_assets) == 1 else None
    if len(cuda_assets) > 1:
        raise ManifestError("manifest contains more than one CUDA runtime")
    if requested == "cpu":
        return RuntimeSelection(cpu, requested, "cpu", False, "已按设置选择 CPU 运行时")
    reason = capability.reason or "未检测到兼容的 NVIDIA CUDA 设备"
    compatible = capability.available and cuda is not None
    if compatible and cuda is not None and cuda.min_nvidia_driver:
        if not capability.driver_version:
            compatible = False
            reason = "无法确定 NVIDIA 驱动版本"
        elif _version_tuple(capability.driver_version) < _version_tuple(cuda.min_nvidia_driver):
            compatible = False
            reason = f"NVIDIA 驱动 {capability.driver_version} 低于要求 {cuda.min_nvidia_driver}"
    if compatible and cuda is not None:
        return RuntimeSelection(cuda, requested, "cuda", False, "已检测到兼容的 NVIDIA CUDA 运行环境")
    if cuda is None:
        reason = "当前版本未提供 CUDA 运行时"
    return RuntimeSelection(cpu, requested, "cpu", requested != "cpu", f"{reason}；已回退 CPU")


def manifest_to_dict(manifest: ReleaseManifest) -> dict[str, Any]:
    """Return a JSON-compatible representation without silently dropping fields."""

    # Round-tripping through the source-like dataclass shape keeps the cached
    # manifest independent from dataclasses.asdict implementation details.
    return json.loads(json.dumps(manifest, default=lambda item: item.__dict__))
