"""Tests for versioned desktop contracts and run storage."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from spimaging.appcore.config import RunConfig
from spimaging.appcore.diagnostics import redact_text
from spimaging.appcore.events import EventType, EventWriter, WorkerEvent
from spimaging.appcore.history import HistoryStore
from spimaging.appcore.specs import (
    RECONSTRUCTION_ALGORITHMS,
    SIMULATION_ALGORITHMS,
    TRAINING_PRESETS,
    validate_training_parameters,
)
from spimaging.appcore.storage import ResultManifest, RunStorage, safe_relative_path


class AlgorithmSpecTests(unittest.TestCase):
    def test_public_algorithm_sets_are_separate_and_complete(self) -> None:
        self.assertEqual(
            set(SIMULATION_ALGORITHMS),
            {"single", "neighborhood_mix", "translucent_layer", "volume_scattering"},
        )
        self.assertEqual(
            set(RECONSTRUCTION_ALGORITHMS),
            {"simple3d", "prsnet", "penonlocal", "stin", "spisr"},
        )
        self.assertTrue(RECONSTRUCTION_ALGORITHMS["simple3d"].bundled_checkpoint)
        self.assertFalse(RECONSTRUCTION_ALGORITHMS["prsnet"].bundled_checkpoint)

    def test_only_effective_model_parameters_are_exposed(self) -> None:
        simple_names = {item.name for item in RECONSTRUCTION_ALGORITHMS["simple3d"].parameters}
        prs_names = {item.name for item in RECONSTRUCTION_ALGORITHMS["prsnet"].parameters}
        stin_names = {item.name for item in RECONSTRUCTION_ALGORITHMS["stin"].parameters}
        self.assertIn("base_channels", simple_names)
        self.assertNotIn("num_blocks", simple_names)
        self.assertIn("num_blocks", prs_names)
        self.assertNotIn("base_channels", prs_names)
        self.assertNotIn("base_channels", stin_names)
        self.assertNotIn("num_blocks", stin_names)

    def test_quick_preset_and_bounds_are_validated(self) -> None:
        values = validate_training_parameters("simple3d", "quick", {"base_channels": 2})
        self.assertEqual(values["epochs"], 1)
        self.assertEqual(values["max_samples"], 2)
        self.assertEqual(values["base_channels"], 2)
        with self.assertRaisesRegex(ValueError, "训练轮数"):
            validate_training_parameters("simple3d", "custom", {"epochs": 0})
        self.assertEqual(TRAINING_PRESETS["standard"]["epochs"], 20)


class RunConfigTests(unittest.TestCase):
    def test_schema_v1_roundtrip_preserves_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "run"
            config = RunConfig.new(
                "full_pipeline",
                run_dir,
                input={"dataset_paths": [str(Path(temporary_directory) / "data")]},
                training={
                    "enabled": True,
                    "model": "simple3d",
                    "preset": "quick",
                    "parameters": {"base_channels": 2},
                },
                visualization={"sample_count": 6, "sample_indices": [0, 2]},
                compute={"preference": "cuda", "gpu_index": 1},
            )
            restored = RunConfig.from_dict(json.loads(config.to_json()))
            self.assertEqual(restored, config)
            self.assertEqual(restored.visualization.sample_count, 6)
            self.assertEqual(restored.compute.gpu_index, 1)

    def test_execution_requirements_and_gallery_bounds_fail_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                RunConfig.new("predict", Path(temporary_directory) / "run")
            with self.assertRaisesRegex(ValueError, "1 到 12"):
                RunConfig.new(
                    "noop",
                    Path(temporary_directory) / "run",
                    visualization={"sample_count": 13},
                )

    def test_unknown_top_level_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = RunConfig.new("noop", Path(temporary_directory) / "run")
            data = config.to_dict()
            data["surprise"] = True
            with self.assertRaisesRegex(ValueError, "未知字段"):
                RunConfig.from_dict(data)


class EventAndStorageTests(unittest.TestCase):
    def test_event_writer_sequences_and_mirrors_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stream = io.StringIO()
            path = Path(temporary_directory) / "events.jsonl"
            writer = EventWriter("run-id", stream=stream, event_path=path)
            first = writer.emit(EventType.STATE, {"status": "running"})
            second = writer.emit(EventType.METRIC, {"mae_m": 0.1})
            self.assertEqual((first.seq, second.seq), (1, 2))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines, stream.getvalue().splitlines())
            self.assertEqual(WorkerEvent.from_dict(json.loads(lines[1])), second)

    def test_run_storage_is_atomic_and_artifacts_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = RunConfig.new("noop", Path(temporary_directory) / "run")
            storage = RunStorage(config)
            manifest = storage.prepare()
            artifact = storage.layout.gallery / "sample.png"
            artifact.write_bytes(b"png")
            manifest.add_artifact("sample", storage.relative_to_root(artifact), "image")
            manifest.set_status("succeeded")
            storage.write_result(manifest)
            restored = storage.load_result()
            self.assertEqual(restored.status, "succeeded")
            self.assertEqual(restored.artifacts[0].path, "gallery/sample.png")
            with self.assertRaises(ValueError):
                safe_relative_path("../outside.txt")
            with self.assertRaisesRegex(ValueError, "运行目录之外"):
                storage.relative_to_root(Path(temporary_directory) / "outside.txt")


class HistoryAndDiagnosticsTests(unittest.TestCase):
    def test_history_marks_interrupted_and_rebuilds_from_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new("noop", root / "runs" / "one")
            storage = RunStorage(config)
            manifest = storage.prepare()
            manifest.set_status("succeeded")
            manifest.metrics = {"ok": True}
            storage.write_result(manifest)
            history = HistoryStore(root / "history.sqlite3")
            history.upsert(config, "running")
            self.assertEqual(history.mark_interrupted(), 1)
            imported, errors = history.rebuild([root / "runs"])
            self.assertEqual((imported, errors), (1, []))
            record = history.list()[0]
            self.assertEqual(record.status, "succeeded")
            self.assertEqual(record.summary, {"ok": True})

    def test_diagnostics_redacts_user_and_explicit_paths(self) -> None:
        secret = Path.home() / "Private Data" / "sample.npz"
        text = f"input={secret}"
        redacted = redact_text(text, [secret.parent])
        self.assertNotIn(str(Path.home()), redacted)
        self.assertNotIn("Private Data", redacted)


if __name__ == "__main__":
    unittest.main()
