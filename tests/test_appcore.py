"""Tests for versioned desktop contracts and run storage."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from spimaging.appcore.config import RunConfig
from spimaging.appcore.diagnostics import export_diagnostic_bundle, redact_text
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
        values = validate_training_parameters("simple3d", "quick", {})
        self.assertEqual(values["epochs"], 1)
        self.assertEqual(values["max_samples"], 2)
        self.assertEqual(values["base_channels"], 2)
        self.assertEqual(values["temporal_downsample"], 64)
        self.assertEqual(
            validate_training_parameters("simple3d", "standard", {})["base_channels"],
            8,
        )
        with self.assertRaisesRegex(ValueError, "训练轮数"):
            validate_training_parameters("simple3d", "custom", {"epochs": 0})
        with self.assertRaisesRegex(ValueError, "奇数"):
            SIMULATION_ALGORITHMS["neighborhood_mix"].validate_parameters(
                {"mix_kernel_size": 4}
            )
        with self.assertRaisesRegex(ValueError, "256"):
            validate_training_parameters("simple3d", "custom", {"base_channels": 257})
        with self.assertRaisesRegex(ValueError, "256"):
            validate_training_parameters(
                "simple3d", "custom", {"base_channels": 10**10000}
            )
        with self.assertRaisesRegex(ValueError, "64"):
            validate_training_parameters("spisr", "custom", {"time_scale": 65})
        self.assertEqual(TRAINING_PRESETS["standard"]["epochs"], 20)
        self.assertEqual(
            validate_training_parameters("spisr", "standard", {})["weight_decay"],
            1e-6,
        )

    def test_volume_defaults_defer_to_medium_and_only_show_effective_boost(self) -> None:
        algorithm = SIMULATION_ALGORITHMS["volume_scattering"]
        defaults = algorithm.parameter_defaults()
        self.assertIsNone(defaults["volume_extinction_coeff"])
        self.assertIsNone(defaults["volume_backscatter_ratio"])
        self.assertIsNone(
            algorithm.validate_parameters({})["volume_extinction_coeff"]
        )

        fog_names = {item.name for item in algorithm.visible_parameters({})}
        water_names = {
            item.name
            for item in algorithm.visible_parameters({"volume_medium_type": "water"})
        }
        self.assertIn("volume_fog_front_boost", fog_names)
        self.assertNotIn("volume_water_front_boost", fog_names)
        self.assertIn("volume_water_front_boost", water_names)
        self.assertNotIn("volume_fog_front_boost", water_names)


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

    def test_schema_types_reject_bool_float_truncation_and_stringification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = RunConfig.new("noop", Path(temporary_directory) / "run").to_dict()
            cases = (
                (("schema_version",), True, "schema_version必须是整数"),
                (("schema_version",), 1.0, "schema_version必须是整数"),
                (("workflow",), 7, "workflow必须是文本"),
                (("display_name",), False, "display_name必须是文本"),
                (("generation", "enabled"), 1, "generation.enabled必须是布尔值"),
                (("generation", "dataset_mode"), 1, "generation.dataset_mode必须是文本"),
                (("training", "preset"), [], "training.preset必须是文本"),
                (("evaluation", "figure_index"), 1.5, "evaluation.figure_index必须是整数"),
                (("visualization", "sample_count"), True, "visualization.sample_count必须是整数"),
                (("visualization", "sample_indices"), [0, 2.5], r"sample_indices\[1\]必须是整数"),
                (("compute", "gpu_index"), "0", "compute.gpu_index必须是整数"),
                (("compute", "preference"), None, "compute.preference必须是文本"),
            )
            for keys, invalid, message in cases:
                data = json.loads(json.dumps(base))
                target = data
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = invalid
                with self.subTest(keys=keys, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, message):
                        RunConfig.from_dict(data)

    def test_full_pipeline_requires_checkpoint_when_training_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                RunConfig.new(
                    "full_pipeline",
                    root / "run",
                    input={"dataset_paths": [str(root / "dataset")]},
                    training={"enabled": False},
                    prediction={"enabled": True},
                    evaluation={"enabled": False},
                )
            valid = RunConfig.new(
                "full_pipeline",
                root / "valid",
                input={
                    "dataset_paths": [str(root / "dataset")],
                    "checkpoint_paths": [str(root / "model.pt")],
                },
                training={"enabled": False},
                prediction={"enabled": True},
                evaluation={"enabled": False},
            )
            self.assertFalse(valid.training.enabled)


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

    def test_event_writer_continues_same_run_without_rewriting_prior_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            first_writer = EventWriter("same-run", event_path=path)
            first_writer.emit(EventType.STATE, {"status": "running"})
            first_writer.emit(EventType.WARNING, {"message": "before restart"})
            original = path.read_bytes()

            second_writer = EventWriter("same-run", event_path=path)
            continued = second_writer.emit(EventType.LOG, {"message": "after restart"})

            self.assertEqual(continued.seq, 3)
            self.assertTrue(path.read_bytes().startswith(original))
            events = [
                WorkerEvent.from_dict(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event.seq for event in events], [1, 2, 3])

    def test_event_writer_discards_only_truncated_tail_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            first = EventWriter("same-run", event_path=path)
            first.emit(EventType.STATE, {"status": "running"})
            complete_prefix = path.read_bytes()
            with path.open("ab") as handle:
                handle.write(b'{"schema_version":1,"run_id":"same-run"')

            second = EventWriter("same-run", event_path=path)
            self.assertEqual(second.emit(EventType.WARNING, {"message": "recovered"}).seq, 2)
            self.assertTrue(path.read_bytes().startswith(complete_prefix))

            third = EventWriter("same-run", event_path=path)
            self.assertEqual(third.emit(EventType.LOG, {"message": "again"}).seq, 3)
            events = [
                WorkerEvent.from_dict(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event.seq for event in events], [1, 2, 3])

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
    def test_corrupt_history_database_is_preserved_and_can_be_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = root / "history.sqlite3"
            database.write_bytes(b"not a sqlite database")

            history = HistoryStore(database)

            self.assertEqual(history.list(), [])
            self.assertIsNotNone(history.recovered_corrupt_path)
            assert history.recovered_corrupt_path is not None
            self.assertEqual(history.recovered_corrupt_path.read_bytes(), b"not a sqlite database")

            config = RunConfig.new("noop", root / "runs" / "one")
            storage = RunStorage(config)
            manifest = storage.prepare()
            manifest.set_status("succeeded")
            storage.write_result(manifest)
            imported, errors = history.rebuild((root / "runs",))
            self.assertEqual((imported, errors), (1, []))
            self.assertEqual(history.list()[0].run_id, config.run_id)

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

    def test_history_recovery_persists_interrupted_authoritative_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new("noop", root / "runs" / "active")
            storage = RunStorage(config)
            manifest = storage.prepare()
            manifest.set_status("running")
            storage.write_result(manifest)
            history = HistoryStore(root / "history.sqlite3")
            history.upsert(config, "running")

            self.assertEqual(history.mark_interrupted(), 1)

            recovered = storage.load_result()
            self.assertEqual(recovered.status, "interrupted")
            self.assertEqual(recovered.error["category"], "UnexpectedInterruption")
            self.assertEqual(history.list()[0].status, "interrupted")

    def test_diagnostics_redacts_user_and_explicit_paths(self) -> None:
        secret = Path.home() / "Private Data" / "sample.npz"
        text = f"input={secret}"
        redacted = redact_text(text, [secret.parent])
        self.assertNotIn(str(Path.home()), redacted)
        self.assertNotIn("Private Data", redacted)

    def test_diagnostic_bundle_redacts_configured_paths_on_other_drives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "run"
            run.mkdir()
            (run / "run.json").write_text(
                json.dumps(
                    {
                        "input": {"dataset_paths": [r"D:\Research Secret\dataset"]},
                        "prediction": {"checkpoint": r"E:\Private Models\best.pt"},
                    }
                ),
                encoding="utf-8",
            )
            (run / "logs").mkdir()
            (run / "logs" / "worker.log").write_text(
                r"reading D:\Research Secret\dataset\sample_00000.npz",
                encoding="utf-8",
            )
            launcher = root / "launcher"
            (launcher / "metadata").mkdir(parents=True)
            (launcher / "metadata" / "update-check.json").write_text(
                json.dumps({"last_check": "2026-08-24T00:00:00+08:00"}),
                encoding="utf-8",
            )

            bundle = export_diagnostic_bundle(
                root / "diagnostics.zip",
                run_dir=run,
                launcher_root=launcher,
            )

            with zipfile.ZipFile(bundle) as archive:
                names = archive.namelist()
                combined = "\n".join(
                    archive.read(name).decode("utf-8") for name in names
                )
            self.assertNotIn("Research Secret", combined)
            self.assertNotIn("Private Models", combined)
            self.assertNotIn("D:\\", combined)
            self.assertTrue(any(name.endswith("update-check.json") for name in names))


if __name__ == "__main__":
    unittest.main()
