"""Persistent partial manifests for cancellable dataset generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


GENERATION_MANIFEST_SCHEMA = 1
PARTIAL_MANIFEST_NAME = ".generation_partial.json"
COMPLETE_MANIFEST_NAME = "generation_manifest.json"
PUBLICATION_JOURNAL_SCHEMA = 1
PUBLICATION_JOURNAL_SUFFIX = ".spimaging-publish.json"
PUBLICATION_BACKUP_SUFFIX = ".spimaging-publish-backup"


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


def publication_journal_for(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    return output_dir.parent / f".{output_dir.name}{PUBLICATION_JOURNAL_SUFFIX}"


def publication_backup_for(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    return output_dir.parent / f".{output_dir.name}{PUBLICATION_BACKUP_SUFFIX}"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
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

    def complete(self, *, retain_partial: bool = False) -> Path:
        completed_manifest = dict(self.manifest)
        completed_manifest["status"] = "complete"
        completed_manifest.pop("interruption_reason", None)
        completed_manifest["sample_count"] = len(completed_manifest["completed"])
        completed_manifest.pop("rng_state", None)
        complete_path = self.directory / COMPLETE_MANIFEST_NAME
        _atomic_write_json(complete_path, completed_manifest)
        partial_path = self.directory / PARTIAL_MANIFEST_NAME
        if not retain_partial and partial_path.exists():
            partial_path.unlink()
            self.manifest = completed_manifest
        return complete_path


def _is_owned_generation_name(name: str) -> bool:
    return name in {
        PARTIAL_MANIFEST_NAME,
        PARTIAL_MANIFEST_NAME + ".tmp",
        COMPLETE_MANIFEST_NAME,
        "index.csv",
    } or (name.startswith("sample_") and name.endswith(".npz"))


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _validate_preserved_source(path: Path) -> None:
    if path.is_symlink():
        raise GenerationResumeError(
            f"refusing to preserve symbolic link during atomic publication: {path.name}"
        )
    if path.is_file():
        return
    if not path.is_dir():
        raise GenerationResumeError(
            f"unsupported existing output entry during publication: {path.name}"
        )
    for root, directory_names, file_names in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in (*directory_names, *file_names):
            candidate = root_path / name
            if candidate.is_symlink():
                raise GenerationResumeError(
                    "refusing to preserve a directory containing symbolic links: "
                    f"{path.name}"
                )


def _copy_preserved_source(source: Path, destination: Path) -> None:
    _validate_preserved_source(source)
    if destination.exists() or destination.is_symlink():
        raise GenerationResumeError(
            f"staging directory already contains preserved output name: {source.name}"
        )
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
    else:
        shutil.copy2(source, destination)


def _publication_paths(output_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    raw_output = Path(output_dir).expanduser()
    raw_staging = partial_directory_for(raw_output)
    if raw_output.is_symlink() or raw_staging.is_symlink():
        raise GenerationResumeError("generation publication paths cannot be symbolic links")
    output = raw_output.resolve()
    staging = raw_staging.resolve()
    if output == staging or output.parent != staging.parent:
        raise GenerationResumeError(
            "generation staging and output directories must be distinct siblings on one volume"
        )
    journal = publication_journal_for(output)
    backup = publication_backup_for(output)
    if journal.parent != output.parent or backup.parent != output.parent:
        raise GenerationResumeError("generation publication metadata escaped the output parent")
    return staging, output, journal, backup


def _load_publication_journal(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 1024 * 1024:
            raise GenerationResumeError("generation publication journal exceeds 1 MiB")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationResumeError(f"cannot read generation publication journal: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != PUBLICATION_JOURNAL_SCHEMA:
        raise GenerationResumeError("generation publication journal schema is invalid")
    if payload.get("phase") not in {"preparing", "prepared", "old_moved", "new_active"}:
        raise GenerationResumeError("generation publication journal phase is invalid")
    if not isinstance(payload.get("had_output"), bool):
        raise GenerationResumeError("generation publication journal output state is invalid")
    preserved = payload.get("preserved")
    if not isinstance(preserved, list) or not all(
        isinstance(name, str)
        and name
        and Path(name).name == name
        and not _is_owned_generation_name(name)
        for name in preserved
    ):
        raise GenerationResumeError("generation publication journal preserved paths are invalid")
    return payload


def _remove_staged_preserved(staging: Path, names: Iterable[str]) -> None:
    for name in names:
        candidate = staging / name
        if candidate.parent != staging:
            raise GenerationResumeError("preserved staging path escaped its directory")
        if candidate.exists() or candidate.is_symlink():
            _remove_path(candidate)


def _is_complete_generation_directory(directory: Path) -> bool:
    manifest_path = directory / COMPLETE_MANIFEST_NAME
    try:
        if manifest_path.stat().st_size > 8 * 1024 * 1024:
            return False
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not (
            isinstance(payload, Mapping)
            and payload.get("schema_version") == GENERATION_MANIFEST_SCHEMA
            and payload.get("status") == "complete"
            and isinstance(payload.get("completed"), list)
            and isinstance(payload.get("sample_count"), int)
            and not isinstance(payload.get("sample_count"), bool)
            and payload.get("sample_count") == len(payload["completed"])
            and (directory / "index.csv").is_file()
        ):
            return False
        listed_samples: set[str] = set()
        for item in payload["completed"]:
            if not isinstance(item, Mapping):
                return False
            name = item.get("file")
            size = item.get("bytes")
            digest = item.get("sha256")
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.startswith("sample_")
                or not name.endswith(".npz")
                or name in listed_samples
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                return False
            sample = directory / name
            if sample.stat().st_size != size or _sha256(sample) != digest:
                return False
            listed_samples.add(name)
        on_disk_samples = {path.name for path in directory.glob("sample_*.npz") if path.is_file()}
        return on_disk_samples == listed_samples
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def recover_generation_publication(output_dir: str | Path) -> str | None:
    """Recover a directory switch interrupted at any Windows-safe rename boundary.

    The transaction journal never supplies filesystem paths; all touched paths
    are derived from the validated output sibling, preventing a forged journal
    from redirecting cleanup outside the generation directory's parent.
    """

    staging, output, journal, backup = _publication_paths(output_dir)
    if not journal.exists():
        if backup.exists() or backup.is_symlink():
            raise GenerationResumeError(
                f"orphaned generation publication backup requires manual inspection: {backup}"
            )
        return None
    if journal.is_symlink() or backup.is_symlink() or output.is_symlink() or staging.is_symlink():
        raise GenerationResumeError("generation publication state cannot contain symbolic links")
    payload = _load_publication_journal(journal)
    preserved = payload["preserved"]

    output_exists = output.is_dir()
    staging_exists = staging.is_dir()
    backup_exists = backup.is_dir()
    if any(
        path.exists() and not path.is_dir()
        for path in (output, staging, backup)
    ):
        raise GenerationResumeError("generation publication state contains a non-directory path")

    # The old directory was renamed but the new directory was not activated.
    # Restore the old canonical path first, then leave the partial generation
    # journal intact so --resume can continue safely.
    if backup_exists and not output_exists:
        os.replace(backup, output)
        output_exists = True
        backup_exists = False
        if staging_exists:
            _remove_staged_preserved(staging, preserved)
        journal.unlink()
        return "old_restored"

    # Once staging has become the canonical output, commit wins.  Windows
    # cannot replace a non-empty directory directly, so the old directory is
    # retained as a rollback backup until this state is unambiguous.
    if output_exists and not staging_exists and _is_complete_generation_directory(output):
        partial = output / PARTIAL_MANIFEST_NAME
        if partial.exists():
            partial.unlink()
        if backup_exists:
            shutil.rmtree(backup)
        journal.unlink()
        return "new_active"

    # No directory switch happened. Remove only the copies named by our own
    # journal and restore a clean resumable staging directory.
    if staging_exists and not backup_exists:
        _remove_staged_preserved(staging, preserved)
        journal.unlink()
        return "staging_resumable"

    raise GenerationResumeError(
        "generation publication state is ambiguous; no directories were modified"
    )


def publish_generation_directory(
    staging_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> None:
    """Publish a complete staging tree using Windows-safe directory renames."""

    expected_staging, output, journal, backup = _publication_paths(output_dir)
    staging = Path(staging_dir).expanduser().resolve()
    if staging != expected_staging:
        raise GenerationResumeError(
            "generation staging directory is not the reserved sibling of the output"
        )
    recover_generation_publication(output)
    if not staging.is_dir() or staging.is_symlink():
        raise GenerationResumeError(f"generation staging directory is missing: {staging}")
    if not _is_complete_generation_directory(staging):
        raise GenerationResumeError("generation staging directory is not complete")
    if output.exists() and not output.is_dir():
        raise GenerationResumeError(f"generation output is not a directory: {output}")
    if output.is_symlink():
        raise GenerationResumeError("generation output cannot be a symbolic link")

    if output.is_dir() and not any(output.iterdir()):
        output.rmdir()
    had_output = output.is_dir()
    if had_output and not overwrite:
        raise GenerationResumeError(f"generation output directory is not empty: {output}")

    preserved_paths = [] if not had_output else [
        path for path in output.iterdir() if not _is_owned_generation_name(path.name)
    ]
    preserved_names = [path.name for path in preserved_paths]
    transaction = {
        "schema_version": PUBLICATION_JOURNAL_SCHEMA,
        "phase": "preparing",
        "had_output": had_output,
        "preserved": preserved_names,
    }
    _atomic_write_json(journal, transaction)
    activated = False
    try:
        for source in preserved_paths:
            _copy_preserved_source(source, staging / source.name)
        transaction["phase"] = "prepared"
        _atomic_write_json(journal, transaction)

        if had_output:
            if backup.exists() or backup.is_symlink():
                raise GenerationResumeError(
                    f"generation publication backup already exists: {backup}"
                )
            os.replace(output, backup)
            transaction["phase"] = "old_moved"
            _atomic_write_json(journal, transaction)

        os.replace(staging, output)
        activated = True
        transaction["phase"] = "new_active"
        _atomic_write_json(journal, transaction)

        partial = output / PARTIAL_MANIFEST_NAME
        if partial.exists():
            partial.unlink()
        if backup.exists():
            shutil.rmtree(backup)
        journal.unlink()
    except Exception as exc:
        # A fault injector (or an unusual filesystem wrapper) can report an
        # error after the rename itself committed. Detect that state from the
        # complete canonical tree; never roll a fully activated new directory
        # back or orphan its old backup merely because the call raised late.
        if output.is_dir() and not staging.exists() and _is_complete_generation_directory(output):
            activated = True
            try:
                recover_generation_publication(output)
                return
            except Exception:
                pass
        if not activated:
            try:
                if backup.is_dir() and not output.exists():
                    os.replace(backup, output)
                if staging.is_dir():
                    _remove_staged_preserved(staging, preserved_names)
                if journal.exists():
                    journal.unlink()
            except Exception:
                # Leave the journal in place. The next invocation will recover
                # the same fixed sibling paths without trusting payload paths.
                pass
        if isinstance(exc, GenerationResumeError):
            raise
        raise GenerationResumeError(f"atomic generation publication failed: {exc}") from exc


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
    recover_generation_publication(output_dir)
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
