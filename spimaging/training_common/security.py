"""Fail-closed loaders for user-provided NumPy and PyTorch files.

The public desktop accepts files selected by the user.  NumPy object arrays
and unrestricted ``torch.load`` both cross a Python-code execution boundary,
so all algorithm entry points share the checks in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path, PurePosixPath
import pickle
from typing import Iterable
import zipfile


MIB = 1024 * 1024


class UnsafeArchiveError(ValueError):
    """Raised when an NPZ archive is malformed or exceeds safety limits."""


class UnsafeCheckpointError(ValueError):
    """Raised when a checkpoint cannot be loaded with the safe schema."""


@dataclass(frozen=True)
class ArchiveLimits:
    """Resource limits applied before an NPZ member is materialized."""

    max_members: int = 128
    max_member_bytes: int = 512 * MIB
    max_total_bytes: int = 1024 * MIB
    max_array_dimensions: int = 5
    max_compression_ratio: float = 2000.0


@dataclass(frozen=True)
class NpzMember:
    key: str
    shape: tuple[int, ...]
    dtype: str
    array_bytes: int
    compressed_bytes: int
    uncompressed_bytes: int


def available_memory_bytes() -> int | None:
    """Best-effort available physical-memory probe without a hard dependency."""

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, OSError, ValueError):
        pass

    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, ValueError):
            pass

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _safe_member_name(filename: str) -> str:
    if "\\" in filename:
        raise UnsafeArchiveError(f"archive member uses a backslash path: {filename!r}")
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise UnsafeArchiveError(f"archive member is outside the archive root: {filename!r}")
    if not filename.endswith(".npy"):
        raise UnsafeArchiveError(f"unexpected non-NPY archive member: {filename!r}")
    key = filename[:-4]
    if not key or len(key) > 128:
        raise UnsafeArchiveError(f"invalid NPZ field name: {key!r}")
    return key


def _read_npy_header(handle):
    import numpy as np

    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(handle)
    if version in {(2, 0), (3, 0)}:
        return np.lib.format.read_array_header_2_0(handle)
    raise UnsafeArchiveError(f"unsupported NPY header version: {version}")


def inspect_npz_archive(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
    limits: ArchiveLimits | None = None,
) -> dict[str, NpzMember]:
    """Inspect an NPZ without loading array payloads into process memory."""

    limits = limits or ArchiveLimits()
    path = Path(path)
    members: dict[str, NpzMember] = {}
    total_bytes = 0
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > limits.max_members:
                raise UnsafeArchiveError(
                    f"archive contains {len(infos)} members; limit is {limits.max_members}"
                )
            for info in infos:
                if info.is_dir():
                    raise UnsafeArchiveError(f"unexpected directory member: {info.filename!r}")
                if info.flag_bits & 0x1:
                    raise UnsafeArchiveError(f"encrypted archive member is not supported: {info.filename!r}")
                key = _safe_member_name(info.filename)
                if key in members:
                    raise UnsafeArchiveError(f"duplicate NPZ field: {key!r}")
                if info.file_size > limits.max_member_bytes:
                    raise UnsafeArchiveError(
                        f"field {key!r} expands to {info.file_size} bytes; "
                        f"per-field limit is {limits.max_member_bytes}"
                    )
                total_bytes += info.file_size
                if total_bytes > limits.max_total_bytes:
                    raise UnsafeArchiveError(
                        f"archive expands to more than {limits.max_total_bytes} bytes"
                    )
                if info.file_size and info.compress_size == 0:
                    raise UnsafeArchiveError(f"field {key!r} has an invalid compressed size")
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > limits.max_compression_ratio:
                    raise UnsafeArchiveError(
                        f"field {key!r} compression ratio {ratio:.1f} exceeds "
                        f"limit {limits.max_compression_ratio:.1f}"
                    )

                try:
                    with archive.open(info, "r") as handle:
                        shape, _fortran_order, dtype = _read_npy_header(handle)
                except (EOFError, OSError, ValueError) as exc:
                    raise UnsafeArchiveError(f"field {key!r} has an invalid NPY header: {exc}") from exc
                if dtype.hasobject:
                    raise UnsafeArchiveError(
                        f"field {key!r} uses object dtype, which would require pickle"
                    )
                if len(shape) > limits.max_array_dimensions:
                    raise UnsafeArchiveError(
                        f"field {key!r} has {len(shape)} dimensions; "
                        f"limit is {limits.max_array_dimensions}"
                    )
                if any(not isinstance(size, int) or size < 0 for size in shape):
                    raise UnsafeArchiveError(f"field {key!r} has an invalid shape: {shape!r}")
                elements = math.prod(shape)
                array_bytes = elements * int(dtype.itemsize)
                if array_bytes > limits.max_member_bytes:
                    raise UnsafeArchiveError(
                        f"field {key!r} needs {array_bytes} array bytes; "
                        f"limit is {limits.max_member_bytes}"
                    )
                if array_bytes > info.file_size:
                    raise UnsafeArchiveError(
                        f"field {key!r} declares more array data than its NPY member contains"
                    )
                members[key] = NpzMember(
                    key=key,
                    shape=tuple(shape),
                    dtype=dtype.str,
                    array_bytes=array_bytes,
                    compressed_bytes=info.compress_size,
                    uncompressed_bytes=info.file_size,
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeArchiveError(f"not a readable NPZ archive: {exc}") from exc

    missing = sorted(set(required_keys) - set(members))
    if missing:
        raise UnsafeArchiveError(f"missing required field(s): {', '.join(missing)}")
    return members


def load_npz_arrays(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
    keys: Iterable[str] | None = None,
    limits: ArchiveLimits | None = None,
) -> dict[str, object]:
    """Eagerly load validated arrays with pickle permanently disabled."""

    metadata = inspect_npz_archive(path, required_keys=required_keys, limits=limits)
    selected = tuple(metadata) if keys is None else tuple(keys)
    missing = sorted(set(selected) - set(metadata))
    if missing:
        raise UnsafeArchiveError(f"missing requested field(s): {', '.join(missing)}")
    requested_bytes = sum(metadata[key].array_bytes for key in selected)
    available_bytes = available_memory_bytes()
    if available_bytes is not None and requested_bytes * 2 > available_bytes:
        raise UnsafeArchiveError(
            f"arrays require {requested_bytes} bytes but only {available_bytes} bytes of "
            "physical memory are currently available; at least 2x headroom is required"
        )

    import numpy as np

    result: dict[str, object] = {}
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            for key in selected:
                value = archive[key]
                if value.dtype.hasobject:
                    raise UnsafeArchiveError(
                        f"field {key!r} uses object dtype, which is not allowed"
                    )
                if value.nbytes != metadata[key].array_bytes:
                    raise UnsafeArchiveError(f"field {key!r} size changed while it was being read")
                result[key] = value
    except UnsafeArchiveError:
        raise
    except (EOFError, KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise UnsafeArchiveError(f"cannot safely load NPZ arrays: {exc}") from exc
    return result


def _numeric_array(data: Mapping[str, object], key: str, dimensions: tuple[int, ...]):
    import numpy as np

    value = data[key]
    if not isinstance(value, np.ndarray) or value.ndim not in dimensions:
        expected = " or ".join(str(item) for item in dimensions)
        raise UnsafeArchiveError(f"field {key!r} must be a {expected}-D NumPy array")
    if value.dtype.kind not in "buif":
        raise UnsafeArchiveError(f"field {key!r} must use a numeric dtype")
    if not np.all(np.isfinite(value)):
        raise UnsafeArchiveError(f"field {key!r} contains NaN or infinite values")
    return value


def validate_spad_sample(
    data: Mapping[str, object],
    *,
    required_keys: Iterable[str] = (),
) -> None:
    """Validate fields consumed by training, prediction, and visualization."""

    import numpy as np

    missing = sorted(set(required_keys) - set(data))
    if missing:
        raise UnsafeArchiveError(f"missing required field(s): {', '.join(missing)}")

    counts = None
    if "counts" in data:
        counts = _numeric_array(data, "counts", (2, 3))
        if not all(size > 0 for size in counts.shape):
            raise UnsafeArchiveError("field 'counts' must not contain empty dimensions")
        if np.any(counts < 0):
            raise UnsafeArchiveError("field 'counts' contains negative photon counts")

    if "depth_m" in data:
        depth = _numeric_array(data, "depth_m", (2,))
        if np.any(depth < 0):
            raise UnsafeArchiveError("field 'depth_m' contains negative depths")
        if counts is not None and tuple(depth.shape) != tuple(counts.shape[-2:]):
            raise UnsafeArchiveError(
                "field 'depth_m' spatial shape must match the last two 'counts' dimensions"
            )

    if "transient_clean" in data:
        transient = _numeric_array(data, "transient_clean", (3,))
        if counts is not None and tuple(transient.shape) != tuple(counts.shape):
            raise UnsafeArchiveError("field 'transient_clean' shape must match 'counts'")
        if np.any(transient < 0):
            raise UnsafeArchiveError("field 'transient_clean' contains negative values")

    if "bin_size" in data:
        value = np.asarray(data["bin_size"])
        if value.ndim != 0 or value.dtype.kind not in "buif":
            raise UnsafeArchiveError("field 'bin_size' must be a numeric scalar")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise UnsafeArchiveError("field 'bin_size' must be finite and greater than zero")


def load_spad_sample(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
    keys: Iterable[str] | None = None,
    limits: ArchiveLimits | None = None,
) -> dict[str, object]:
    data = load_npz_arrays(path, required_keys=required_keys, keys=keys, limits=limits)
    validate_spad_sample(data, required_keys=required_keys)
    return data


def load_checkpoint_safely(path: str | Path, *, map_location="cpu") -> Mapping[str, object]:
    """Load a tensor checkpoint without allowing arbitrary pickle globals."""

    import torch

    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise UnsafeCheckpointError(f"cannot inspect checkpoint: {exc}") from exc
    if size > 2 * 1024 * MIB:
        raise UnsafeCheckpointError("checkpoint exceeds the 2 GiB safety limit")
    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as exc:
        raise UnsafeCheckpointError(
            "this PyTorch build does not support safe weights_only checkpoint loading"
        ) from exc
    except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise UnsafeCheckpointError(f"safe checkpoint load failed: {exc}") from exc

    if not isinstance(checkpoint, Mapping):
        raise UnsafeCheckpointError("checkpoint root must be a mapping")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise UnsafeCheckpointError("checkpoint field 'model_state' is missing or invalid")
    if any(not isinstance(key, str) for key in model_state):
        raise UnsafeCheckpointError("checkpoint model_state keys must be strings")
    train_args = checkpoint.get("args", {})
    if not isinstance(train_args, Mapping) or any(not isinstance(key, str) for key in train_args):
        raise UnsafeCheckpointError("checkpoint field 'args' must be a string-keyed mapping")
    if "optimizer_state" in checkpoint and not isinstance(checkpoint["optimizer_state"], Mapping):
        raise UnsafeCheckpointError("checkpoint field 'optimizer_state' must be a mapping")
    if "epoch" in checkpoint and (
        isinstance(checkpoint["epoch"], bool)
        or not isinstance(checkpoint["epoch"], int)
        or checkpoint["epoch"] < 0
    ):
        raise UnsafeCheckpointError("checkpoint field 'epoch' must be a nonnegative integer")
    for key in ("model_name", "method_family"):
        if key in checkpoint and not isinstance(checkpoint[key], str):
            raise UnsafeCheckpointError(f"checkpoint field {key!r} must be a string")
    return checkpoint
