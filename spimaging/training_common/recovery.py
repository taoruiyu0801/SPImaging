"""Training history, deterministic fingerprints, and resume validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from spimaging.training_common.security import load_checkpoint_safely


RESUME_SCHEMA_VERSION = 1


class IncompatibleResumeError(ValueError):
    """Raised when a checkpoint does not describe the requested experiment."""


def dataset_fingerprint(files: Iterable[str | Path]) -> str:
    """Fingerprint ordered file content without storing personal paths."""

    digest = hashlib.sha256()
    for value in files:
        path = Path(value)
        stat = path.stat()
        record = {
            "name": path.name,
            "size": stat.st_size,
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def build_resume_signature(args: object, keys: Iterable[str]) -> dict[str, object]:
    return {key: getattr(args, key) for key in keys}


def capture_rng_state() -> dict[str, object]:
    """Capture RNG state using only values accepted by weights-only loading."""

    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].astype("uint32").tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    import numpy as np
    import torch

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(tuple(python_state))
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, Mapping):
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    torch_cpu = state.get("torch_cpu")
    if not isinstance(torch_cpu, torch.Tensor):
        raise IncompatibleResumeError("checkpoint CPU RNG state is missing or invalid")
    if torch_cpu.dtype is not torch.uint8 or torch_cpu.ndim != 1:
        raise IncompatibleResumeError("checkpoint CPU RNG tensor has an invalid shape or dtype")
    torch.set_rng_state(torch_cpu.detach().cpu().contiguous())
    torch_cuda = state.get("torch_cuda")
    if torch_cuda is not None and torch.cuda.is_available():
        if not isinstance(torch_cuda, (list, tuple)):
            raise IncompatibleResumeError("checkpoint CUDA RNG state is invalid")
        for index, cuda_state in enumerate(torch_cuda[: torch.cuda.device_count()]):
            if (
                not isinstance(cuda_state, torch.Tensor)
                or cuda_state.dtype is not torch.uint8
                or cuda_state.ndim != 1
            ):
                raise IncompatibleResumeError("checkpoint CUDA RNG tensor is invalid")
            torch.cuda.set_rng_state(
                cuda_state.detach().cpu().contiguous(),
                device=index,
            )


def build_resume_metadata(
    *,
    dataset_hash: str,
    signature: Mapping[str, object],
    target_epochs: int,
    next_epoch: int,
    next_batch: int,
    global_step: int,
    phase: str = "train",
) -> dict[str, object]:
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_hash,
        "signature": dict(signature),
        "target_epochs": int(target_epochs),
        "next_epoch": int(next_epoch),
        "next_batch": int(next_batch),
        "global_step": int(global_step),
        "phase": str(phase),
        "rng_state": capture_rng_state(),
    }


def load_and_validate_resume(
    path: str | Path,
    *,
    map_location,
    dataset_hash: str,
    signature: Mapping[str, object],
    requested_epochs: int,
) -> Mapping[str, object]:
    checkpoint = load_checkpoint_safely(path, map_location=map_location)
    metadata = checkpoint.get("resume")
    if not isinstance(metadata, Mapping):
        raise IncompatibleResumeError("checkpoint has no resumable state metadata")
    if metadata.get("schema_version") != RESUME_SCHEMA_VERSION:
        raise IncompatibleResumeError("checkpoint uses an unsupported resume schema")
    if metadata.get("dataset_fingerprint") != dataset_hash:
        raise IncompatibleResumeError("dataset fingerprint does not match the checkpoint")
    saved_signature = metadata.get("signature")
    if not isinstance(saved_signature, Mapping) or dict(saved_signature) != dict(signature):
        raise IncompatibleResumeError("algorithm, network, or preprocessing parameters changed")
    original_target = metadata.get("target_epochs")
    if not isinstance(original_target, int) or isinstance(original_target, bool):
        raise IncompatibleResumeError("checkpoint target epoch metadata is invalid")
    if int(requested_epochs) < original_target:
        raise IncompatibleResumeError(
            f"--epochs may stay at {original_target} or increase when resuming, but cannot decrease"
        )
    next_epoch = metadata.get("next_epoch")
    next_batch = metadata.get("next_batch")
    if (
        not isinstance(next_epoch, int)
        or isinstance(next_epoch, bool)
        or next_epoch < 1
        or not isinstance(next_batch, int)
        or isinstance(next_batch, bool)
        or next_batch < 0
    ):
        raise IncompatibleResumeError("checkpoint resume position is invalid")
    if not isinstance(checkpoint.get("optimizer_state"), Mapping):
        raise IncompatibleResumeError("checkpoint has no optimizer state")
    return checkpoint


def restore_checkpoint_state(model, optimizer, checkpoint: Mapping[str, object]) -> dict[str, object]:
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise IncompatibleResumeError(f"checkpoint tensor or optimizer state is incompatible: {exc}") from exc
    metadata = dict(checkpoint["resume"])
    rng_state = metadata.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise IncompatibleResumeError("checkpoint RNG state is missing or invalid")
    try:
        restore_rng_state(rng_state)
    except IncompatibleResumeError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise IncompatibleResumeError(f"checkpoint RNG state is incompatible: {exc}") from exc
    return metadata


HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_kl",
    "train_tv",
    "train_mae_m",
    "train_pukl",
    "train_equivariance",
    "val_loss",
    "val_kl",
    "val_tv",
    "val_mae_m",
    "val_pukl",
    "val_equivariance",
    "global_step",
)


def append_training_history(output_dir: str | Path, row: Mapping[str, Any]) -> None:
    """Append one epoch to JSONL and CSV machine-readable histories."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = {key: row.get(key, "") for key in HISTORY_FIELDS}
    json_path = output_dir / "training_history.jsonl"
    with json_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")

    csv_path = output_dir / "training_history.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)
