"""Offline structural/hash/extraction dry-run for a release manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from launcher.archive import inspect_zip  # noqa: E402
from launcher.download import sha256_file  # noqa: E402
from launcher.manifest import ReleaseManifest  # noqa: E402


def verify(manifest_path: Path, asset_dir: Path | None) -> ReleaseManifest:
    manifest = ReleaseManifest.from_json(manifest_path.read_bytes())
    if asset_dir is None:
        return manifest
    asset_dir = asset_dir.resolve(strict=True)
    for asset in manifest.assets:
        archive = asset_dir / asset.archive_name
        if not archive.is_file() or archive.stat().st_size != asset.archive_size:
            raise ValueError(f"missing or wrong-size archive: {archive}")
        if sha256_file(archive) != asset.archive_sha256:
            raise ValueError(f"archive hash mismatch: {archive}")
        for part in asset.parts:
            part_path = asset_dir / part.name
            if not part_path.is_file() or part_path.stat().st_size != part.size or sha256_file(part_path) != part.sha256:
                raise ValueError(f"part validation failed: {part_path}")
        with tempfile.TemporaryDirectory() as temporary:
            inspect_zip(archive, Path(temporary), max_unpacked_size=asset.unpacked_size)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--asset-dir", type=Path)
    args = parser.parse_args()
    manifest = verify(args.manifest, args.asset_dir)
    print(f"OK schema={manifest.schema_version} version={manifest.release_version} assets={len(manifest.assets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
