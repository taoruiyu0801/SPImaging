"""Shared command-line parsing, validation, and path helpers."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable


HelpFormatter = argparse.ArgumentDefaultsHelpFormatter


class ArgumentParser(argparse.ArgumentParser):
    """Argument parser with concise, consistent user-facing errors."""

    def error(self, message: str) -> None:
        self.exit(
            2,
            f"{self.prog}: error: {message}\n"
            f"Run '{self.prog} --help' for usage.\n",
        )


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a finite number")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def random_seed(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def fraction(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number < 1:
        raise argparse.ArgumentTypeError("must satisfy 0 <= value < 1")
    return number


def unit_interval(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def positive_unit_interval(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must satisfy 0 < value <= 1")
    return number


def parameter_index(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 10:
        raise argparse.ArgumentTypeError("must be between 1 and 10")
    return number


def positive_odd_int(value: str) -> int:
    number = positive_int(value)
    if number % 2 == 0:
        raise argparse.ArgumentTypeError("must be an odd integer")
    return number


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared explicit device controls without importing PyTorch."""

    device_default = os.environ.get("SPIMAGING_DEVICE", "auto").strip().lower()
    if device_default not in {"auto", "cuda", "cpu"}:
        device_default = "auto"
    gpu_default = os.environ.get("SPIMAGING_GPU_INDEX", "0").strip()
    try:
        gpu_default = str(nonnegative_int(gpu_default))
    except (argparse.ArgumentTypeError, ValueError):
        gpu_default = "0"
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=device_default,
        help="Execution device; CUDA requests fall back to CPU with an explanation when unavailable.",
    )
    parser.add_argument(
        "--gpu_index",
        type=nonnegative_int,
        default=gpu_default,
        help="Zero-based NVIDIA GPU index used by auto/cuda device modes.",
    )


def require_directory(parser: argparse.ArgumentParser, value: str, option: str) -> Path:
    path = Path(value)
    if not path.exists():
        parser.error(f"{option} directory does not exist: {path}")
    if not path.is_dir():
        parser.error(f"{option} must be a directory: {path}")
    return path


def require_file(
    parser: argparse.ArgumentParser,
    value: str,
    option: str,
    suffixes: Iterable[str] = (),
) -> Path:
    path = Path(value)
    if not path.exists():
        parser.error(f"{option} file does not exist: {path}")
    if not path.is_file():
        parser.error(f"{option} must be a file: {path}")
    allowed = tuple(suffix.lower() for suffix in suffixes)
    if allowed and path.suffix.lower() not in allowed:
        choices = ", ".join(allowed)
        parser.error(f"{option} must use one of these extensions: {choices}")
    return path


def require_dataset_path(parser: argparse.ArgumentParser, value: str, option: str) -> Path:
    path = Path(value)
    if not path.exists():
        parser.error(f"{option} path does not exist: {path}")
    if path.is_dir():
        return path
    if path.is_file() and path.suffix.lower() == ".npz":
        return path
    parser.error(f"{option} must be a directory or an .npz file: {path}")


def validate_npz_archive(
    parser: argparse.ArgumentParser,
    path: Path,
    option: str,
    required_keys: Iterable[str] = (),
) -> Path:
    from spimaging.training_common.security import UnsafeArchiveError, inspect_npz_archive

    try:
        inspect_npz_archive(path, required_keys=required_keys)
    except UnsafeArchiveError as exc:
        parser.error(f"{option} is not a safe readable .npz archive: {path} ({exc})")
    return path


def validate_output_directory(
    parser: argparse.ArgumentParser,
    value: str,
    *,
    overwrite: bool,
    option: str = "--output_dir",
    require_empty: bool = True,
) -> Path:
    path = Path(value)
    if path.exists() and not path.is_dir():
        parser.error(f"{option} must be a directory: {path}")
    if path.is_dir() and require_empty and not overwrite:
        try:
            next(path.iterdir())
        except StopIteration:
            pass
        except OSError as exc:
            parser.error(f"cannot inspect {option} {path}: {exc}")
        else:
            parser.error(f"{option} is not empty: {path}; use --overwrite to allow existing output")
    return path


def create_output_directory(
    parser: argparse.ArgumentParser,
    path: Path,
    *,
    option: str = "--output_dir",
) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(f"cannot create {option} {path}: {exc}")
    return path


def validate_output_file(
    parser: argparse.ArgumentParser,
    value: str,
    *,
    overwrite: bool,
    option: str,
    suffixes: Iterable[str] = (),
) -> Path:
    path = Path(value)
    allowed = tuple(suffix.lower() for suffix in suffixes)
    if allowed and path.suffix.lower() not in allowed:
        choices = ", ".join(allowed)
        parser.error(f"{option} must use one of these extensions: {choices}")
    if path.exists() and path.is_dir():
        parser.error(f"{option} must be a file path: {path}")
    if path.exists() and not overwrite:
        parser.error(f"{option} already exists: {path}; use --overwrite to replace it")
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        parser.error(f"parent path for {option} is not a directory: {parent}")
    return path


def create_output_parent(
    parser: argparse.ArgumentParser,
    path: Path,
    *,
    option: str,
) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(f"cannot create parent directory for {option} {path}: {exc}")
    return path
