"""Fail-closed extraction for downloaded ZIP release assets."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import os
import shutil
import stat
import zipfile

from .errors import ExtractionError


DEFAULT_MAX_MEMBERS = 200_000
DEFAULT_MAX_COMPRESSION_RATIO = 1_000
_WINDOWS_DEVICE_NAMES = {
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
    *(f"COM{index}" for index in "¹²³"),
    *(f"LPT{index}" for index in "¹²³"),
}


def _member_target(destination: Path, name: str) -> Path:
    if not name or "\\" in name or "\x00" in name or ":" in name or "//" in name:
        raise ExtractionError(f"unsafe archive member path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ExtractionError(f"unsafe archive member path: {name!r}")
    for part in relative.parts:
        stem = part.rstrip(". ").split(".", 1)[0].upper()
        if part.endswith((".", " ")) or stem in _WINDOWS_DEVICE_NAMES:
            raise ExtractionError(f"unsafe Windows archive member path: {name!r}")
    target = destination.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError as error:
        raise ExtractionError(f"archive member escapes staging directory: {name!r}") from error
    return target


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(unix_mode)


def inspect_zip(
    archive: Path,
    destination: Path,
    *,
    max_unpacked_size: int,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_compression_ratio: int = DEFAULT_MAX_COMPRESSION_RATIO,
) -> tuple[list[zipfile.ZipInfo], int]:
    """Validate every member before the first byte is extracted."""

    try:
        handle = zipfile.ZipFile(archive, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ExtractionError(f"invalid ZIP archive {archive.name}: {error}") from error
    with handle:
        infos = handle.infolist()
        if not infos or len(infos) > max_members:
            raise ExtractionError(f"ZIP member count must be between 1 and {max_members}")
        total = 0
        names: set[str] = set()
        for info in infos:
            _member_target(destination, info.filename)
            folded = info.filename.rstrip("/").casefold()
            if folded in names:
                raise ExtractionError(f"duplicate archive member path: {info.filename!r}")
            names.add(folded)
            if info.flag_bits & 0x1:
                raise ExtractionError(f"encrypted ZIP members are unsupported: {info.filename!r}")
            if _is_symlink(info):
                raise ExtractionError(f"symbolic links are forbidden in release archives: {info.filename!r}")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(unix_mode)
            if file_type and not (stat.S_ISREG(unix_mode) or stat.S_ISDIR(unix_mode)):
                raise ExtractionError(f"special file is forbidden in release archive: {info.filename!r}")
            total += info.file_size
            if total > max_unpacked_size:
                raise ExtractionError("ZIP expands beyond the manifest's unpacked_size")
            if info.file_size and info.compress_size == 0:
                raise ExtractionError(f"invalid zero compressed size: {info.filename!r}")
            if info.compress_size and info.file_size / info.compress_size > max_compression_ratio:
                raise ExtractionError(f"suspicious compression ratio: {info.filename!r}")
        free = shutil.disk_usage(destination.parent).free
        margin = max(64 * 1024 * 1024, total // 20)
        if free < total + margin:
            raise ExtractionError(f"insufficient disk space to extract archive: need {total + margin}, have {free}")
        return infos, total


def safe_extract_zip(
    archive: Path,
    destination: Path,
    *,
    max_unpacked_size: int,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> int:
    """Extract regular files/directories only, after validating the entire ZIP."""

    if destination.exists() and any(destination.iterdir()):
        raise ExtractionError("staging directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    infos, total = inspect_zip(
        archive,
        destination,
        max_unpacked_size=max_unpacked_size,
        max_members=max_members,
    )
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            for info in infos:
                target = _member_target(destination, info.filename)
                if info.is_dir() or info.filename.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ExtractionError(f"ZIP extraction failed: {error}") from error
    return total
