"""Day 23 pytest-style checks for invalid user inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from spimaging.cli import ArgumentParser, parameter_index, require_directory
from spimaging.generation.pipeline import build_parser


def test_missing_input_path_raises_without_creating_output(tmp_path: Path) -> None:
    parser = ArgumentParser(prog="day23-missing-path")
    missing = tmp_path / "missing-input"
    output = tmp_path / "must-not-exist"

    with pytest.raises(SystemExit) as raised:
        require_directory(parser, str(missing), "--dataset_dir")

    assert raised.value.code == 2
    assert not output.exists()


def test_invalid_sbr_preset_index_raises() -> None:
    """SBR is selected through --param_idx rather than a public --sbr option."""

    with pytest.raises(argparse.ArgumentTypeError):
        parameter_index("0")
    with pytest.raises(argparse.ArgumentTypeError):
        parameter_index("11")


def test_invalid_surface_model_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    parser = build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["--surface_model", "not-a-model"])

    assert raised.value.code == 2
    assert list(tmp_path.iterdir()) == []
