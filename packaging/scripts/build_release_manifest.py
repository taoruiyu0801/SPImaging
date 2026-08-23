"""Create a validated release manifest from already-built local assets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from launcher.manifest import ReleaseManifest  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def parts_for(archive: Path, base_url: str) -> list[dict[str, object]]:
    record_path = archive.parent / f"{archive.name}.parts.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("archive_size") != archive.stat().st_size or record.get("archive_sha256") != digest(archive):
            raise ValueError(f"stale split-part record for {archive.name}")
        raw_parts = record.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise ValueError(f"invalid split-part record for {archive.name}")
        result = []
        for part in raw_parts:
            part_path = archive.parent / part["name"]
            if not part_path.is_file() or part_path.stat().st_size != part["size"] or digest(part_path) != part["sha256"]:
                raise ValueError(f"split part does not match record: {part_path.name}")
            result.append({**part, "url": f"{base_url.rstrip('/')}/{quote(part_path.name)}"})
        return result
    return [
        {
            "name": archive.name,
            "url": f"{base_url.rstrip('/')}/{quote(archive.name)}",
            "size": archive.stat().st_size,
            "sha256": digest(archive),
        }
    ]


def asset(
    archive: Path,
    *,
    version: str,
    component: str,
    variant: str,
    base_url: str,
    unpacked_size: int,
    signature: dict[str, object],
    min_driver: str | None = None,
) -> dict[str, object]:
    archive = archive.resolve(strict=True)
    required_paths = (
        ["python.exe", "Scripts/conda-unpack.exe"]
        if component == "runtime"
        else ["spimaging/__init__.py"]
    )
    result: dict[str, object] = {
        "asset_id": f"spimaging-{component}-{variant}-{version}",
        "component": component,
        "variant": variant,
        "platform": "windows-x86_64",
        "version": version,
        "archive_name": archive.name,
        "archive_format": "zip",
        "archive_size": archive.stat().st_size,
        "archive_sha256": digest(archive),
        "unpacked_size": unpacked_size,
        "parts": parts_for(archive, base_url),
        "required_paths": required_paths,
        "relocation": (
            {
                "executable": "Scripts/conda-unpack.exe",
                "arguments": [],
                "expected_exit_code": 0,
                "timeout_seconds": 180,
            }
            if component == "runtime"
            else None
        ),
        "health_check": (
            {
                "executable": "python.exe",
                "arguments": [
                    "-c",
                    (
                        "import torch,sys;print(torch.__version__);"
                        "import PySide6,deepinv;"
                        + ("sys.exit(0 if torch.cuda.is_available() else 3)" if variant == "cuda" else "sys.exit(0)")
                    ),
                ],
                "expected_exit_code": 0,
                "timeout_seconds": 90,
            }
            if component == "runtime"
            else None
        ),
        "signature": signature,
    }
    if min_driver:
        result["min_nvidia_driver"] = min_driver
    return result


def signature_policy(args: argparse.Namespace, archive: Path) -> dict[str, object]:
    if args.unsigned_beta:
        return {"kind": "none", "required": False}
    if not args.signer_thumbprint:
        raise ValueError("signed manifest requires --signer-thumbprint")
    signature_file = archive.with_suffix(archive.suffix + ".p7s")
    if not signature_file.is_file() or signature_file.stat().st_size == 0:
        raise ValueError(f"detached CMS signature is missing: {signature_file}")
    return {
        "kind": "cms-detached",
        "required": True,
        "signer_thumbprint": args.signer_thumbprint,
        "file_name": signature_file.name,
        "url": f"{args.base_url.rstrip('/')}/{quote(signature_file.name)}",
        "size": signature_file.stat().st_size,
        "sha256": digest(signature_file),
    }


def unpacked_size(archive: Path) -> int:
    import zipfile

    with zipfile.ZipFile(archive, "r") as handle:
        return sum(item.file_size for item in handle.infolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--runtime-version",
        help="independently versioned CPU/CUDA layer (defaults to --version)",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cpu-archive", type=Path, required=True)
    parser.add_argument("--cuda-archive", type=Path)
    parser.add_argument("--app-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unsigned-beta", action="store_true")
    parser.add_argument("--signer-thumbprint")
    parser.add_argument("--min-nvidia-driver", default="550.0")
    args = parser.parse_args()
    runtime_version = args.runtime_version or args.version
    archives = [args.cpu_archive, args.app_archive] + ([args.cuda_archive] if args.cuda_archive else [])
    policies = {path: signature_policy(args, path) for path in archives}
    assets = [
        asset(
            args.cpu_archive,
            version=runtime_version,
            component="runtime",
            variant="cpu",
            base_url=args.base_url,
            unpacked_size=unpacked_size(args.cpu_archive),
            signature=policies[args.cpu_archive],
        ),
        asset(
            args.app_archive,
            version=args.version,
            component="app",
            variant="universal",
            base_url=args.base_url,
            unpacked_size=unpacked_size(args.app_archive),
            signature=policies[args.app_archive],
        ),
    ]
    if args.cuda_archive:
        assets.append(
            asset(
                args.cuda_archive,
                version=runtime_version,
                component="runtime",
                variant="cuda",
                base_url=args.base_url,
                unpacked_size=unpacked_size(args.cuda_archive),
                signature=policies[args.cuda_archive],
                min_driver=args.min_nvidia_driver,
            )
        )
    manifest = {
        "schema_version": 1,
        "product": "SPImaging",
        "release_version": args.version,
        "channel": "beta" if "-" in args.version else "stable",
        "published_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "unsigned_beta": args.unsigned_beta,
        "assets": assets,
        "launch": {
            "runtime_executable": "pythonw.exe",
            "console_executable": "python.exe",
            "app_module": "spimaging.desktop",
            "arguments": [],
        },
    }
    validated = ReleaseManifest.from_dict(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"validated schema v{validated.schema_version}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
