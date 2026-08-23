"""Build a deterministic source asset without private/NYUv2-derived data."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
ALLOWED_ROOTS = {"spimaging", "public_demo"}
REQUIRED_FILES = {"LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md", "SBOM.md"}
FORBIDDEN_PARTS = {"example_data", "demo_checkpoint", "record_of_SPI", ".git"}


def tracked_files(repo: Path) -> list[str]:
    completed = subprocess.run(
        [
            "git", "ls-files", "-z", "--",
            "spimaging", "public_demo", "LICENSE", "NOTICE",
            "THIRD_PARTY_LICENSES.md", "SBOM.md",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    names = [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]
    return sorted(names)


def validate_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe app asset path: {name}")
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError(f"private path is forbidden in app asset: {name}")
    if path.parts[0] not in ALLOWED_ROOTS | REQUIRED_FILES:
        raise ValueError(f"path is outside the app allowlist: {name}")
    return path


def build(repo: Path, output: Path) -> None:
    repo = repo.resolve(strict=True)
    names = tracked_files(repo)
    if not REQUIRED_FILES.issubset(names):
        missing = ", ".join(sorted(REQUIRED_FILES.difference(names)))
        raise ValueError(f"required public compliance files must be tracked before building: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            path = validate_name(name)
            source = repo.joinpath(*path.parts).resolve(strict=True)
            source.relative_to(repo)
            if not source.is_file():
                continue
            info = zipfile.ZipInfo(path.as_posix(), FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
