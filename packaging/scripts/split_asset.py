"""Split a release asset into deterministic, GitHub-friendly named parts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os


DEFAULT_PART_SIZE = 1_800 * 1024 * 1024


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def split_asset(source: Path, output_dir: Path, part_size: int = DEFAULT_PART_SIZE) -> Path:
    source = source.resolve(strict=True)
    if not source.is_file() or part_size < 1024 * 1024:
        raise ValueError("source must be a file and part_size must be at least 1 MiB")
    output_dir.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, object]] = []
    with source.open("rb") as handle:
        index = 1
        while chunk := handle.read(part_size):
            name = f"{source.name}.{index:03d}"
            path = output_dir / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as output:
                output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            parts.append({"name": name, "size": len(chunk), "sha256": digest(path)})
            index += 1
    if not parts:
        raise ValueError("cannot split an empty asset")
    record = {
        "schema_version": 1,
        "archive_name": source.name,
        "archive_size": source.stat().st_size,
        "archive_sha256": digest(source),
        "parts": parts,
    }
    manifest = output_dir / f"{source.name}.parts.json"
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    args = parser.parse_args()
    print(split_asset(args.source, args.output_dir, args.part_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
