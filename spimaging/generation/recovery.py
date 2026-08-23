"""Persistent partial manifests for cancellable dataset generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


GENERATION_MANIFEST_SCHEMA = 1
PARTIAL_MANIFEST_NAME = ".generation_partial.json"
COMPLETE_MANIFEST_NAME = "generation_manifest.json"


class GenerationResumeError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generation_config_fingerprint(args: object) -> str:
    excluded = {"output_dir", "overwrite", "resume"}
    values = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and not key.startswith("_")
    }
    return _stable_hash(values)


def source_fingerprint(paths: Iterable[str | Path]) -> str:
    records = []
    for value in paths:
        path = Path(value)
        stat = path.stat()
        records.append(
            {
                "path_hash": hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest(),
                "name": path.name,
                "size": stat.st_size,
                "sha256": _sha256(path),
            }
        )
    return _stable_hash(records)


def partial_directory_for(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    return output_dir.parent / f".{output_dir.name}.spimaging-partial"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _numpy_rng_to_json() -> dict[str, object]:
    import numpy as np

    state = np.random.get_state()
    return {
        "algorithm": state[0],
        "keys": state[1].astype("uint32").tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_numpy_rng(state: Mapping[str, object]) -> None:
    import numpy as np

    np.random.set_state(
        (
            str(state["algorithm"]),
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["position"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


class GenerationSession:
    def __init__(self, directory: Path, manifest: dict[str, Any]) -> None:
        self.directory = directory
        self.manifest = manifest

    @property
    def completed_ids(self) -> set[int]:
        return {int(item["sample_id"]) for item in self.manifest["completed"]}

    @property
    def index_rows(self) -> list[dict[str, object]]:
        return [dict(item["index_row"]) for item in self.manifest["completed"]]

    def record_sample(self, sample_id: int, sample_path: Path, index_row: Mapping[str, object]) -> None:
        completed = [
            item for item in self.manifest["completed"] if int(item["sample_id"]) != int(sample_id)
        ]
        completed.append(
            {
                "sample_id": int(sample_id),
                "file": sample_path.name,
                "bytes": sample_path.stat().st_size,
                "sha256": _sha256(sample_path),
                "index_row": dict(index_row),
            }
        )
        completed.sort(key=lambda item: int(item["sample_id"]))
        self.manifest["completed"] = completed
        self.manifest["rng_state"] = _numpy_rng_to_json()
        self.manifest["status"] = "incomplete"
        _atomic_write_json(self.directory / PARTIAL_MANIFEST_NAME, self.manifest)

    def mark_interrupted(self, reason: str) -> None:
        self.manifest["status"] = "incomplete"
        self.manifest["interruption_reason"] = str(reason)
        self.manifest["rng_state"] = _numpy_rng_to_json()
        _atomic_write_json(self.directory / PARTIAL_MANIFEST_NAME, self.manifest)

    def complete(self) -> Path:
        self.manifest["status"] = "complete"
        self.manifest.pop("interruption_reason", None)
        self.manifest["sample_count"] = len(self.manifest["completed"])
        self.manifest.pop("rng_state", None)
        complete_path = self.directory / COMPLETE_MANIFEST_NAME
        _atomic_write_json(complete_path, self.manifest)
        partial_path = self.directory / PARTIAL_MANIFEST_NAME
        if partial_path.exists():
            partial_path.unlink()
        return complete_path


def _clear_owned_partial(directory: Path) -> None:
    allowed_names = {PARTIAL_MANIFEST_NAME, PARTIAL_MANIFEST_NAME + ".tmp", COMPLETE_MANIFEST_NAME, "index.csv"}
    unexpected = [
        path.name
        for path in directory.iterdir()
        if not (path.is_file() and (path.name in allowed_names or path.name.startswith("sample_") and path.suffix == ".npz"))
    ]
    if unexpected:
        raise GenerationResumeError(
            "partial directory contains unowned entries and will not be replaced: " + ", ".join(unexpected)
        )
    for path in list(directory.iterdir()):
        if path.is_file():
            path.unlink()
    directory.rmdir()


def prepare_generation_session(
    output_dir: str | Path,
    *,
    config_hash: str,
    source_hash: str,
    total_candidates: int,
    resume: bool,
    overwrite: bool,
) -> GenerationSession:
    directory = partial_directory_for(output_dir)
    manifest_path = directory / PARTIAL_MANIFEST_NAME

    if directory.exists() and not directory.is_dir():
        raise GenerationResumeError(f"partial generation path is not a directory: {directory}")
    if directory.is_dir() and resume:
        try:
            if manifest_path.stat().st_size > 8 * 1024 * 1024:
                raise GenerationResumeError("partial manifest exceeds the 8 MiB safety limit")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GenerationResumeError(f"cannot read partial manifest: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != GENERATION_MANIFEST_SCHEMA:
            raise GenerationResumeError("partial manifest schema is invalid")
        if manifest.get("status") != "incomplete":
            raise GenerationResumeError("partial manifest is not resumable")
        if manifest.get("config_fingerprint") != config_hash:
            raise GenerationResumeError("generation parameters differ from the partial run")
        if manifest.get("source_fingerprint") != source_hash:
            raise GenerationResumeError("source data differs from the partial run")
        if int(manifest.get("total_candidates", -1)) != int(total_candidates):
            raise GenerationResumeError("source candidate count differs from the partial run")
        completed = manifest.get("completed")
        if not isinstance(completed, list):
            raise GenerationResumeError("partial manifest completed list is invalid")
        for item in completed:
            if not isinstance(item, dict):
                raise GenerationResumeError("partial manifest sample record is invalid")
            sample = directory / str(item.get("file", ""))
            if sample.parent != directory or not sample.is_file():
                raise GenerationResumeError(f"partial sample is missing: {sample.name}")
            if sample.stat().st_size != item.get("bytes") or _sha256(sample) != item.get("sha256"):
                raise GenerationResumeError(f"partial sample failed integrity validation: {sample.name}")
        rng_state = manifest.get("rng_state")
        if isinstance(rng_state, Mapping):
            try:
                _restore_numpy_rng(rng_state)
            except (KeyError, TypeError, ValueError) as exc:
                raise GenerationResumeError(f"partial RNG state is invalid: {exc}") from exc
        return GenerationSession(directory, manifest)

    if directory.is_dir():
        if not overwrite:
            raise GenerationResumeError(
                f"an incomplete generation exists at {directory}; use --resume to continue or --overwrite to replace it"
            )
        _clear_owned_partial(directory)
    elif resume:
        raise GenerationResumeError(f"no incomplete generation exists for --output_dir {output_dir}")

    directory.mkdir(parents=False, exist_ok=False)
    manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA,
        "status": "incomplete",
        "config_fingerprint": config_hash,
        "source_fingerprint": source_hash,
        "total_candidates": int(total_candidates),
        "completed": [],
        "rng_state": _numpy_rng_to_json(),
    }
    _atomic_write_json(manifest_path, manifest)
    return GenerationSession(directory, manifest)


def cleanup_completed_session(session: GenerationSession) -> None:
    """Remove the now-empty sibling staging directory after publication."""

    if session.directory.exists():
        remaining = list(session.directory.iterdir())
        if remaining:
            raise GenerationResumeError(
                "completed partial directory still contains files: "
                + ", ".join(path.name for path in remaining)
            )
        session.directory.rmdir()
