"""T05 security, checkpoint, history, and partial-generation tests."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from spimaging.generation.recovery import (
    COMPLETE_MANIFEST_NAME,
    GenerationResumeError,
    generation_config_fingerprint,
    prepare_generation_session,
    source_fingerprint,
)
from spimaging.training_common.device import select_torch_device
from spimaging.training_common.recovery import (
    IncompatibleResumeError,
    append_training_history,
    build_resume_metadata,
    dataset_fingerprint,
    load_and_validate_resume,
    restore_rng_state,
)
from spimaging.training_common.security import (
    ArchiveLimits,
    UnsafeArchiveError,
    UnsafeCheckpointError,
    inspect_npz_archive,
    load_npz_arrays,
    load_checkpoint_safely,
    load_spad_sample,
)
from spimaging.training_common.utils import save_training_checkpoint


def _sample(path: Path, *, include_depth: bool = True) -> Path:
    values = {
        "counts": np.ones((4, 3, 2), dtype=np.float32),
        "bin_size": np.asarray(80e-12, dtype=np.float32),
    }
    if include_depth:
        values["depth_m"] = np.ones((3, 2), dtype=np.float32)
    np.savez_compressed(path, **values)
    return path


def test_object_dtype_npz_is_rejected_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "object.npz"
    np.savez(path, counts=np.asarray([{"untrusted": True}], dtype=object))

    with pytest.raises(UnsafeArchiveError, match="object dtype"):
        inspect_npz_archive(path)


def test_npz_member_path_and_resource_limits_are_checked(tmp_path: Path) -> None:
    payload = io.BytesIO()
    np.save(payload, np.ones((2,), dtype=np.float32), allow_pickle=False)
    traversal = tmp_path / "traversal.npz"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../counts.npy", payload.getvalue())
    with pytest.raises(UnsafeArchiveError, match="outside the archive root"):
        inspect_npz_archive(traversal)

    valid = _sample(tmp_path / "valid.npz")
    with pytest.raises(UnsafeArchiveError, match="expands"):
        inspect_npz_archive(valid, limits=ArchiveLimits(max_total_bytes=32))


def test_spad_schema_rejects_shape_mismatch_and_loads_valid_input(tmp_path: Path) -> None:
    valid = _sample(tmp_path / "valid.npz")
    loaded = load_spad_sample(valid, required_keys=("counts", "depth_m"))
    assert loaded["counts"].shape == (4, 3, 2)

    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(
        invalid,
        counts=np.ones((4, 3, 2), dtype=np.float32),
        depth_m=np.ones((2, 3), dtype=np.float32),
    )
    with pytest.raises(UnsafeArchiveError, match="spatial shape"):
        load_spad_sample(invalid, required_keys=("counts", "depth_m"))


def test_npz_load_checks_current_available_memory(tmp_path: Path, monkeypatch) -> None:
    valid = _sample(tmp_path / "valid.npz")
    monkeypatch.setattr(
        "spimaging.training_common.security.available_memory_bytes",
        lambda: 8,
    )
    with pytest.raises(UnsafeArchiveError, match="physical memory"):
        load_npz_arrays(valid)


def test_weights_only_checkpoint_blocks_untrusted_globals(tmp_path: Path) -> None:
    import torch

    marker = tmp_path / "must-not-exist.txt"

    class Payload:
        def __reduce__(self):
            return (Path.write_text, (marker, "executed"))

    checkpoint = tmp_path / "untrusted.pt"
    torch.save({"model_state": {"weight": torch.ones(1)}, "payload": Payload()}, checkpoint)

    with pytest.raises(UnsafeCheckpointError, match="safe checkpoint load failed"):
        load_checkpoint_safely(checkpoint)
    assert not marker.exists()


def test_device_selection_is_explicit_and_explains_fallback(monkeypatch) -> None:
    import torch

    cpu = select_torch_device("cpu", 7)
    assert str(cpu.device) == "cpu"
    assert not cpu.fallback

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    automatic = select_torch_device("auto", 0)
    assert str(automatic.device) == "cpu"
    assert automatic.fallback
    assert "CUDA is not available" in automatic.reason

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        select_torch_device("cuda", 0)


def test_resume_metadata_validates_dataset_signature_and_epoch_direction(tmp_path: Path) -> None:
    import torch

    sample = _sample(tmp_path / "sample.npz")
    dataset_hash = dataset_fingerprint([sample])
    signature = {"model": "simple3d", "base_channels": 1}
    args = argparse.Namespace(model="simple3d", base_channels=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    resume = build_resume_metadata(
        dataset_hash=dataset_hash,
        signature=signature,
        target_epochs=3,
        next_epoch=2,
        next_batch=1,
        global_step=4,
    )
    path = tmp_path / "resume.pt"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        1,
        args,
        "simple3d",
        "supervised",
        0.5,
        "best_val_mae",
        resume=resume,
    )

    loaded = load_and_validate_resume(
        path,
        map_location="cpu",
        dataset_hash=dataset_hash,
        signature=signature,
        requested_epochs=3,
    )
    assert loaded["resume"]["next_batch"] == 1

    with pytest.raises(IncompatibleResumeError, match="cannot decrease"):
        load_and_validate_resume(
            path,
            map_location="cpu",
            dataset_hash=dataset_hash,
            signature=signature,
            requested_epochs=2,
        )
    with pytest.raises(IncompatibleResumeError, match="fingerprint"):
        load_and_validate_resume(
            path,
            map_location="cpu",
            dataset_hash="different",
            signature=signature,
            requested_epochs=3,
        )


def test_supervised_resume_signature_includes_early_stopping_policy() -> None:
    from spimaging.supervised_training.train import SUPERVISED_RESUME_SIGNATURE_FIELDS

    assert "early_stopping_patience" in SUPERVISED_RESUME_SIGNATURE_FIELDS
    assert "early_stopping_min_delta" in SUPERVISED_RESUME_SIGNATURE_FIELDS


def test_rng_restore_forces_tensor_state_to_cpu_and_rejects_invalid_values(monkeypatch) -> None:
    import torch

    captured = []
    state = torch.get_rng_state()
    monkeypatch.setattr(torch, "set_rng_state", lambda value: captured.append(value))

    restore_rng_state({"torch_cpu": state})

    assert len(captured) == 1
    assert captured[0].device.type == "cpu"
    assert captured[0].dtype == torch.uint8
    with pytest.raises(IncompatibleResumeError, match="CPU RNG"):
        restore_rng_state({"torch_cpu": [1, 2, 3]})


def test_training_history_writes_jsonl_and_csv(tmp_path: Path) -> None:
    append_training_history(tmp_path, {"epoch": 1, "train_loss": 0.25, "val_loss": 0.5, "global_step": 2})
    append_training_history(tmp_path, {"epoch": 2, "train_loss": 0.2, "val_loss": 0.4, "global_step": 4})

    records = [json.loads(line) for line in (tmp_path / "training_history.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2]
    csv_lines = (tmp_path / "training_history.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 3


def test_generation_partial_manifest_resumes_and_never_claims_early_success(tmp_path: Path) -> None:
    source = _sample(tmp_path / "source.npz")
    output = tmp_path / "published"
    args = argparse.Namespace(output_dir=str(output), overwrite=False, resume=False, surface_model="single")
    config_hash = generation_config_fingerprint(args)
    source_hash = source_fingerprint([source])
    session = prepare_generation_session(
        output,
        config_hash=config_hash,
        source_hash=source_hash,
        total_candidates=2,
        resume=False,
        overwrite=False,
    )
    generated = _sample(session.directory / "sample_00000.npz")
    session.record_sample(0, generated, {"sample_id": 0, "file": generated.name})
    session.mark_interrupted("cancelled")

    partial = json.loads((session.directory / ".generation_partial.json").read_text(encoding="utf-8"))
    assert partial["status"] == "incomplete"
    assert not (session.directory / COMPLETE_MANIFEST_NAME).exists()

    resumed = prepare_generation_session(
        output,
        config_hash=config_hash,
        source_hash=source_hash,
        total_candidates=2,
        resume=True,
        overwrite=False,
    )
    assert resumed.completed_ids == {0}
    complete_path = resumed.complete()
    assert json.loads(complete_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert not (resumed.directory / ".generation_partial.json").exists()


def test_generation_resume_rejects_tampered_partial_sample(tmp_path: Path) -> None:
    source = _sample(tmp_path / "source.npz")
    output = tmp_path / "published"
    session = prepare_generation_session(
        output,
        config_hash="config",
        source_hash=source_fingerprint([source]),
        total_candidates=1,
        resume=False,
        overwrite=False,
    )
    generated = _sample(session.directory / "sample_00000.npz")
    session.record_sample(0, generated, {"sample_id": 0, "file": generated.name})
    generated.write_bytes(b"tampered")

    with pytest.raises(GenerationResumeError, match="integrity"):
        prepare_generation_session(
            output,
            config_hash="config",
            source_hash=source_fingerprint([source]),
            total_candidates=1,
            resume=True,
            overwrite=False,
        )
