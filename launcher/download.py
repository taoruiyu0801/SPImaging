"""Range-resuming, hash-verified release asset downloads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import os
import shutil
from typing import BinaryIO, Callable, Mapping, Protocol
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .errors import DownloadError, SignatureError
from .manifest import AssetPart, ReleaseAsset
from .signing import SignatureVerifier


ProgressCallback = Callable[[str, int, int], None]


@dataclass
class DownloadResponse:
    stream: BinaryIO
    status: int
    headers: Mapping[str, str]

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> "DownloadResponse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class DownloadTransport(Protocol):
    def open(self, url: str, start: int = 0) -> DownloadResponse: ...


class UrllibTransport:
    """HTTPS transport using the OS/Python proxy configuration."""

    def __init__(self, timeout_seconds: int = 60) -> None:
        self.timeout_seconds = timeout_seconds

    def open(self, url: str, start: int = 0) -> DownloadResponse:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise DownloadError("launcher network requests require HTTPS URLs without credentials")
        headers = {"User-Agent": "SPImaging-Launcher/0.2"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout_seconds)
        except (urllib.error.URLError, OSError) as error:
            raise DownloadError(f"download request failed for {url}: {error}") from error
        final_url = response.geturl()
        if urlparse(final_url).scheme != "https":
            response.close()
            raise DownloadError("refusing a download redirected away from HTTPS")
        return DownloadResponse(response, getattr(response, "status", response.getcode()), response.headers)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, expected_size: int, expected_sha256: str) -> bool:
    try:
        return path.is_file() and path.stat().st_size == expected_size and sha256_file(path) == expected_sha256
    except OSError:
        return False


def _content_range_starts_at(headers: Mapping[str, str], start: int) -> bool:
    value = headers.get("Content-Range") or headers.get("content-range") or ""
    return value.lower().startswith(f"bytes {start}-")


def download_part(
    part: AssetPart,
    destination: Path,
    transport: DownloadTransport,
    progress: ProgressCallback | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download one part, preserving a trustworthy partial file for resume."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if validate_file(destination, part.size, part.sha256):
        if progress:
            progress(part.name, part.size, part.size)
        return destination
    if destination.exists():
        destination.unlink()
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink():
        partial.unlink()
    start = partial.stat().st_size if partial.is_file() else 0
    if start > part.size:
        partial.unlink()
        start = 0
    try:
        response = transport.open(part.url, start)
    except DownloadError:
        raise
    except Exception as error:
        raise DownloadError(f"download request failed for {part.name}: {error}") from error
    with response:
        if response.status not in {200, 206}:
            raise DownloadError(f"server returned HTTP {response.status} for {part.name}")
        append = start > 0 and response.status == 206
        if append and not _content_range_starts_at(response.headers, start):
            raise DownloadError(f"server returned an invalid Content-Range for {part.name}")
        if start > 0 and response.status == 200:
            start = 0
            append = False
        mode = "ab" if append else "wb"
        downloaded = start
        try:
            with partial.open(mode) as output:
                while chunk := response.stream.read(chunk_size):
                    downloaded += len(chunk)
                    if downloaded > part.size:
                        raise DownloadError(f"download exceeded declared size for {part.name}")
                    output.write(chunk)
                    if progress:
                        progress(part.name, downloaded, part.size)
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            raise DownloadError(f"could not write {part.name}: {error}") from error
    actual_size = partial.stat().st_size if partial.is_file() else 0
    if actual_size < part.size:
        raise DownloadError(f"incomplete download for {part.name}: received {actual_size} of {part.size} bytes")
    if not validate_file(partial, part.size, part.sha256):
        partial.unlink(missing_ok=True)
        raise DownloadError(f"SHA-256 or size mismatch for {part.name}")
    os.replace(partial, destination)
    return destination


def _ensure_disk_space(directory: Path, required: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    # Keep a small margin for state files and filesystem accounting.
    margin = max(64 * 1024 * 1024, required // 20)
    if free < required + margin:
        raise DownloadError(f"insufficient disk space: need {required + margin} bytes, have {free}")


def _signature_part(asset: ReleaseAsset) -> AssetPart | None:
    requirement = asset.signature
    if requirement.kind != "cms-detached":
        return None
    assert requirement.file_name and requirement.url and requirement.size and requirement.sha256
    return AssetPart(requirement.file_name, requirement.url, requirement.size, requirement.sha256)


def download_asset(
    asset: ReleaseAsset,
    cache_root: Path,
    transport: DownloadTransport,
    verifier: SignatureVerifier | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download/assemble an asset, then enforce its hash and signature policy."""

    asset_cache = cache_root / asset.asset_id
    asset_cache.mkdir(parents=True, exist_ok=True)
    try:
        asset_cache.resolve().relative_to(cache_root.resolve())
    except ValueError as error:
        raise DownloadError("asset cache path escapes the managed cache root") from error
    _ensure_disk_space(asset_cache, asset.archive_size * 2 + (asset.signature.size or 0))
    downloaded = [download_part(part, asset_cache / part.name, transport, progress) for part in asset.parts]
    archive = asset_cache / asset.archive_name
    if not validate_file(archive, asset.archive_size, asset.archive_sha256):
        archive.unlink(missing_ok=True)
        assembling = archive.with_name(archive.name + ".assembling")
        assembling.unlink(missing_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with assembling.open("xb") as output:
                for part_path in downloaded:
                    with part_path.open("rb") as source:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            assembling.unlink(missing_ok=True)
            raise DownloadError(f"could not assemble {asset.archive_name}: {error}") from error
        if size != asset.archive_size or digest.hexdigest() != asset.archive_sha256:
            assembling.unlink(missing_ok=True)
            raise DownloadError(f"assembled SHA-256 or size mismatch for {asset.archive_name}")
        os.replace(assembling, archive)
    signature_path = None
    signature_part = _signature_part(asset)
    if signature_part is not None:
        signature_path = download_part(signature_part, asset_cache / signature_part.name, transport, progress)
    if asset.signature.required:
        if verifier is None:
            raise SignatureError(f"{asset.asset_id} requires a publisher signature verifier")
        verifier.verify(archive, asset.signature, signature_path)
    return archive


def fetch_bytes(
    url: str,
    transport: DownloadTransport,
    *,
    max_bytes: int = 2 * 1024 * 1024,
) -> bytes:
    """Fetch a small control file with a strict upper bound."""

    try:
        response = transport.open(url, 0)
    except DownloadError:
        raise
    except Exception as error:
        raise DownloadError(f"request failed for {url}: {error}") from error
    result = bytearray()
    with response:
        if response.status != 200:
            raise DownloadError(f"server returned HTTP {response.status} for {url}")
        while chunk := response.stream.read(64 * 1024):
            result.extend(chunk)
            if len(result) > max_bytes:
                raise DownloadError(f"response exceeded {max_bytes} bytes")
    return bytes(result)
