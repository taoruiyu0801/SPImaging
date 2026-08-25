"""Crash-boundary tests for atomic generated-dataset publication."""

from __future__ import annotations

from pathlib import Path

import pytest

import spimaging.generation.recovery as recovery


class SimulatedHardCrash(BaseException):
    """Bypass normal rollback exactly like abrupt process termination."""


def _complete_staging(output: Path) -> Path:
    session = recovery.prepare_generation_session(
        output,
        config_hash="config",
        source_hash="source",
        total_candidates=1,
        resume=False,
        overwrite=False,
    )
    sample = session.directory / "sample_00000.npz"
    sample.write_bytes(b"new complete sample")
    session.record_sample(0, sample, {"sample_id": 0, "file": sample.name})
    (session.directory / "index.csv").write_text("new index", encoding="utf-8")
    session.complete(retain_partial=True)
    return session.directory


def _old_output(output: Path) -> None:
    output.mkdir()
    (output / "sample_00000.npz").write_bytes(b"old complete sample")
    (output / "index.csv").write_text("old index", encoding="utf-8")
    (output / "keep.txt").write_text("preserve me", encoding="utf-8")


def test_crash_after_old_directory_move_restores_old_and_keeps_resume_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dataset"
    _old_output(output)
    staging = _complete_staging(output)
    backup = recovery.publication_backup_for(output)
    real_replace = recovery.os.replace

    def crash_after_old_move(source, destination):
        result = real_replace(source, destination)
        if Path(source) == output and Path(destination) == backup:
            raise SimulatedHardCrash("power loss after old directory move")
        return result

    monkeypatch.setattr(recovery.os, "replace", crash_after_old_move)
    with pytest.raises(SimulatedHardCrash):
        recovery.publish_generation_directory(staging, output, overwrite=True)
    monkeypatch.setattr(recovery.os, "replace", real_replace)

    assert not output.exists()
    assert backup.is_dir()
    assert (staging / recovery.PARTIAL_MANIFEST_NAME).is_file()

    assert recovery.recover_generation_publication(output) == "old_restored"
    assert (output / "sample_00000.npz").read_bytes() == b"old complete sample"
    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserve me"
    assert staging.is_dir()
    assert (staging / recovery.PARTIAL_MANIFEST_NAME).is_file()
    assert not backup.exists()
    assert not recovery.publication_journal_for(output).exists()


def test_crash_after_new_directory_activation_commits_whole_new_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dataset"
    _old_output(output)
    staging = _complete_staging(output)
    backup = recovery.publication_backup_for(output)
    real_replace = recovery.os.replace

    def crash_after_new_activation(source, destination):
        result = real_replace(source, destination)
        if Path(source) == staging and Path(destination) == output:
            raise SimulatedHardCrash("power loss after new directory activation")
        return result

    monkeypatch.setattr(recovery.os, "replace", crash_after_new_activation)
    with pytest.raises(SimulatedHardCrash):
        recovery.publish_generation_directory(staging, output, overwrite=True)
    monkeypatch.setattr(recovery.os, "replace", real_replace)

    assert output.is_dir()
    assert backup.is_dir()
    assert not staging.exists()
    assert (output / recovery.PARTIAL_MANIFEST_NAME).is_file()

    assert recovery.recover_generation_publication(output) == "new_active"
    assert (output / "sample_00000.npz").read_bytes() == b"new complete sample"
    assert (output / "index.csv").read_text(encoding="utf-8") == "new index"
    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserve me"
    assert not (output / recovery.PARTIAL_MANIFEST_NAME).exists()
    assert not backup.exists()
    assert not recovery.publication_journal_for(output).exists()


def test_late_replace_error_does_not_roll_back_an_observably_complete_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dataset"
    _old_output(output)
    staging = _complete_staging(output)
    real_replace = recovery.os.replace

    def error_after_new_activation(source, destination):
        result = real_replace(source, destination)
        if Path(source) == staging and Path(destination) == output:
            raise OSError("injected error after committed rename")
        return result

    monkeypatch.setattr(recovery.os, "replace", error_after_new_activation)
    recovery.publish_generation_directory(staging, output, overwrite=True)
    monkeypatch.setattr(recovery.os, "replace", real_replace)

    assert (output / "sample_00000.npz").read_bytes() == b"new complete sample"
    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserve me"
    assert not recovery.publication_backup_for(output).exists()
    assert not recovery.publication_journal_for(output).exists()


def test_integrity_failure_never_moves_the_existing_complete_output(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    _old_output(output)
    staging = _complete_staging(output)
    (staging / "sample_00000.npz").write_bytes(b"tampered after completion")

    with pytest.raises(recovery.GenerationResumeError, match="not complete"):
        recovery.publish_generation_directory(staging, output, overwrite=True)

    assert (output / "sample_00000.npz").read_bytes() == b"old complete sample"
    assert (staging / recovery.PARTIAL_MANIFEST_NAME).is_file()
    assert not recovery.publication_backup_for(output).exists()
    assert not recovery.publication_journal_for(output).exists()
