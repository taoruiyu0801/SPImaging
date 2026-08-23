"""Tests for the isolated schema-v1 worker."""

from __future__ import annotations

import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from spimaging.appcore.config import RunConfig
from spimaging.appcore.events import EventType
from spimaging.appcore.storage import ResultManifest
from spimaging.appcore.storage import RunStorage
from spimaging.worker import WorkerRuntime, build_parser
from spimaging.worker import _completed_generation_is_valid
from spimaging.generation.recovery import partial_directory_for


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class WorkerRuntimeTests(unittest.TestCase):
    def test_dataset_resolution_honors_workflow_specific_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evaluation_dataset = root / "requested evaluation data"
            evaluation_dataset.mkdir()
            evaluation = RunConfig.new(
                "evaluate",
                root / "evaluate run",
                evaluation={
                    "checkpoints": [str(root / "model.pt")],
                    "dataset_dir": str(evaluation_dataset),
                },
            )
            self.assertEqual(
                WorkerRuntime(evaluation, stream=io.StringIO())._resolve_dataset(),
                evaluation_dataset.resolve(),
            )

            sample = root / "single sample" / "sample.npz"
            prediction = RunConfig.new(
                "predict",
                root / "predict run",
                prediction={
                    "checkpoint": str(root / "model.pt"),
                    "sample_file": str(sample),
                },
            )
            self.assertEqual(
                WorkerRuntime(prediction, stream=io.StringIO())._resolve_dataset(),
                sample.parent.resolve(),
            )

    def test_full_pipeline_routes_training_and_evaluation_datasets_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_dataset = root / "training A"
            evaluation_dataset = root / "evaluation B"
            training_dataset.mkdir()
            evaluation_dataset.mkdir()
            checkpoint = root / "model.pt"
            config = RunConfig.new(
                "full_pipeline",
                root / "run",
                input={"dataset_paths": [str(training_dataset)]},
                training={"enabled": True},
                prediction={"enabled": False},
                evaluation={
                    "enabled": True,
                    "checkpoints": [str(checkpoint)],
                    "dataset_dir": str(evaluation_dataset),
                },
                output={"history_db": str(root / "history.sqlite3")},
            )
            runtime = WorkerRuntime(config, stream=io.StringIO())
            runtime.run_command = mock.Mock()

            self.assertEqual(runtime.execute(), 0)

            commands = {
                call.args[0]: call.args[1]
                for call in runtime.run_command.call_args_list
            }
            self.assertEqual(
                Path(commands["inspect"][commands["inspect"].index("--dataset_dir") + 1]),
                training_dataset.resolve(),
            )
            self.assertEqual(
                Path(commands["train"][commands["train"].index("--dataset_dir") + 1]),
                training_dataset.resolve(),
            )
            self.assertEqual(
                Path(commands["evaluate"][commands["evaluate"].index("--dataset_dir") + 1]),
                evaluation_dataset.resolve(),
            )

    def test_generation_adapter_uses_resume_only_for_an_existing_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new(
                "generate",
                root / "run",
                input={"source_path": str(root / "source")},
                generation={
                    "enabled": True,
                    "dataset_mode": "middlebury",
                    "surface_model": "neighborhood_mix",
                    "resume": True,
                },
                output={"history_db": str(root / "history.sqlite3")},
            )
            runtime = WorkerRuntime(config, stream=io.StringIO())
            output_dir = runtime.layout.artifacts / "dataset"

            fresh = runtime._generation_command(output_dir)
            self.assertNotIn("--resume", fresh)
            self.assertNotIn("--overwrite", fresh)

            partial_directory_for(output_dir).mkdir(parents=True)
            resumed = runtime._generation_command(output_dir)
            self.assertIn("--resume", resumed)
            self.assertNotIn("--overwrite", resumed)

    def test_generation_adapter_replaces_owned_state_only_when_resume_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new(
                "generate",
                root / "run",
                input={"source_path": str(root / "source")},
                generation={
                    "enabled": True,
                    "dataset_mode": "middlebury",
                    "surface_model": "neighborhood_mix",
                    "resume": False,
                },
                output={"history_db": str(root / "history.sqlite3")},
            )
            runtime = WorkerRuntime(config, stream=io.StringIO())
            output_dir = runtime.layout.artifacts / "dataset"
            partial_directory_for(output_dir).mkdir(parents=True)

            command = runtime._generation_command(output_dir)

            self.assertIn("--overwrite", command)
            self.assertNotIn("--resume", command)

    def test_volume_adapter_omits_nullable_defaults_and_irrelevant_boost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new(
                "generate",
                root / "run",
                input={"source_path": str(root / "source")},
                generation={
                    "enabled": True,
                    "dataset_mode": "raw",
                    "surface_model": "volume_scattering",
                },
            )
            command = WorkerRuntime(config, stream=io.StringIO())._generation_command(
                root / "run" / "artifacts" / "dataset"
            )

            self.assertNotIn("--volume_extinction_coeff", command)
            self.assertNotIn("--volume_backscatter_ratio", command)
            self.assertIn("--volume_fog_front_boost", command)
            self.assertNotIn("--volume_water_front_boost", command)

            overridden = RunConfig.new(
                "generate",
                root / "override",
                input={"source_path": str(root / "source")},
                generation={
                    "enabled": True,
                    "dataset_mode": "raw",
                    "surface_model": "volume_scattering",
                    "parameters": {
                        "volume_medium_type": "water",
                        "volume_extinction_coeff": 0.3,
                    },
                },
            )
            override_command = WorkerRuntime(
                overridden, stream=io.StringIO()
            )._generation_command(root / "override" / "artifacts" / "dataset")
            self.assertIn("--volume_extinction_coeff", override_command)
            self.assertIn("--volume_water_front_boost", override_command)
            self.assertNotIn("--volume_fog_front_boost", override_command)

    def test_completed_generation_is_reused_only_after_integrity_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "dataset"
            output_dir.mkdir()
            sample = output_dir / "sample_00000.npz"
            sample.write_bytes(b"synthetic sample")
            digest = hashlib.sha256(sample.read_bytes()).hexdigest()
            (output_dir / "generation_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "completed": [
                            {
                                "file": sample.name,
                                "bytes": sample.stat().st_size,
                                "sha256": digest,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_completed_generation_is_valid(output_dir))
            sample.write_bytes(b"tampered")
            self.assertFalse(_completed_generation_is_valid(output_dir))

    def test_training_resume_reuses_original_generated_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_run = root / "original"
            original = RunConfig.new(
                "full_pipeline",
                original_run,
                input={"source_path": str(root / "source")},
                generation={"enabled": True},
                training={"enabled": True},
                prediction={"enabled": False},
                evaluation={"enabled": False},
            )
            original_storage = RunStorage(original)
            original_storage.prepare()
            dataset = original_storage.layout.artifacts / "dataset"
            dataset.mkdir()
            sample = dataset / "sample_00000.npz"
            sample.write_bytes(b"stable generated sample")
            (dataset / "generation_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "completed": [
                            {
                                "file": sample.name,
                                "bytes": sample.stat().st_size,
                                "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = original_storage.layout.checkpoints / "cancelled.pt"
            checkpoint.write_bytes(b"checkpoint")

            resumed = RunConfig.new(
                "full_pipeline",
                root / "resumed",
                input={"source_path": str(root / "source")},
                generation={"enabled": True},
                training={"enabled": True, "resume_checkpoint": str(checkpoint)},
                prediction={"enabled": False},
                evaluation={"enabled": False},
                output={"history_db": str(root / "history.sqlite3")},
            )
            runtime = WorkerRuntime(resumed, stream=io.StringIO())
            runtime.run_command = mock.Mock()

            self.assertEqual(runtime.execute(), 0)

            stages = [call.args[0] for call in runtime.run_command.call_args_list]
            self.assertEqual(stages, ["inspect", "train"])
            train_command = runtime.run_command.call_args_list[1].args[1]
            dataset_index = train_command.index("--dataset_dir") + 1
            self.assertEqual(Path(train_command[dataset_index]), dataset.resolve())

    def test_same_run_generation_restart_keeps_partial_and_event_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new(
                "generate",
                root / "run",
                input={"source_path": str(root / "source")},
                generation={"enabled": True, "resume": True},
                output={"history_db": str(root / "history.sqlite3")},
            )
            storage = RunStorage(config)
            storage.prepare()
            prior_runtime = WorkerRuntime(config, stream=io.StringIO())
            prior_runtime.events.emit(EventType.STATE, {"status": "cancelled"})
            partial_directory_for(storage.layout.artifacts / "dataset").mkdir()

            stream = io.StringIO()
            restarted = WorkerRuntime(config, stream=stream)
            restarted.run_command = mock.Mock()

            self.assertEqual(restarted.execute(), 0)
            generation_command = restarted.run_command.call_args.args[1]
            self.assertIn("--resume", generation_command)
            new_events = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(new_events[0]["seq"], 2)
            all_events = [
                json.loads(line)
                for line in storage.layout.events.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["seq"] for event in all_events],
                list(range(1, len(all_events) + 1)),
            )

    def test_noop_workflow_writes_complete_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new(
                "noop",
                root / "run with spaces",
                output={"history_db": str(root / "history.sqlite3")},
            )
            stream = io.StringIO()
            runtime = WorkerRuntime(config, stream=stream)

            returncode = runtime.execute()

            self.assertEqual(returncode, 0)
            result = runtime.storage.load_result()
            self.assertEqual(result.status, "succeeded")
            self.assertTrue(runtime.layout.config.is_file())
            self.assertTrue(runtime.layout.events.is_file())
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(events[0]["type"], EventType.STATE.value)
            self.assertEqual(events[-1]["payload"]["status"], "succeeded")
            self.assertEqual([event["seq"] for event in events], list(range(1, len(events) + 1)))

    def test_terminal_fallback_preserves_primary_error_across_cleanup_and_io_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new(
                "noop",
                root / "run",
                output={"history_db": str(root / "history.sqlite3")},
            )
            runtime = WorkerRuntime(config, stream=io.StringIO())
            runtime._run_noop = mock.Mock(
                side_effect=RuntimeError("primary training failure")
            )
            runtime._collect_results = mock.Mock(
                side_effect=OSError("disk full while collecting results")
            )

            real_write_result = runtime.storage.write_result

            def fail_normal_terminal_write(manifest):
                if manifest.status in {"failed", "cancelled", "succeeded"}:
                    raise OSError("disk full during terminal manifest write")
                return real_write_result(manifest)

            runtime.storage.write_result = mock.Mock(
                side_effect=fail_normal_terminal_write
            )
            real_history_upsert = runtime.history.upsert

            def fail_terminal_history(config_value, status, summary=None):
                if status in {"failed", "cancelled", "succeeded"}:
                    raise OSError("history database unavailable")
                return real_history_upsert(config_value, status, summary)

            runtime.history.upsert = mock.Mock(side_effect=fail_terminal_history)
            real_emit = runtime.events.emit

            def fail_primary_terminal_emitter(event_type, payload=None):
                payload = dict(payload or {})
                name = event_type.value if isinstance(event_type, EventType) else str(event_type)
                if name in {"error", "completed"} or payload.get("status") in {
                    "failed",
                    "cancelled",
                    "succeeded",
                }:
                    raise OSError("primary event writer unavailable")
                return real_emit(event_type, payload)

            runtime.events.emit = mock.Mock(side_effect=fail_primary_terminal_emitter)

            real_replace = os.replace

            def fail_emergency_atomic_replace(source, destination):
                if ".terminal-" in Path(source).name:
                    raise OSError("terminal atomic replace unavailable")
                return real_replace(source, destination)

            with mock.patch(
                "spimaging.worker.os.replace",
                side_effect=fail_emergency_atomic_replace,
            ):
                self.assertEqual(runtime.execute(), 1)

            manifest = ResultManifest.from_dict(
                json.loads(runtime.layout.result.read_text(encoding="utf-8"))
            )
            self.assertEqual(manifest.status, "failed")
            self.assertEqual(manifest.error["category"], "RuntimeError")
            self.assertEqual(manifest.error["message"], "primary training failure")
            self.assertTrue(
                any("collect_results" in item for item in manifest.error["secondary_errors"])
            )
            self.assertTrue(any("紧急精简清单" in warning for warning in manifest.warnings))
            self.assertFalse(runtime._terminal_reserve.exists())

            events = [
                json.loads(line)
                for line in runtime.layout.events.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[-1]["type"], "completed")
            self.assertEqual(events[-1]["payload"]["status"], "failed")
            self.assertTrue(any(event["type"] == "error" for event in events))
            self.assertEqual(
                [event["seq"] for event in events],
                list(range(1, len(events) + 1)),
            )

    def test_terminal_fallback_refuses_non_authoritative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new("noop", root / "run")
            runtime = WorkerRuntime(config, stream=io.StringIO())

            with self.assertRaisesRegex(ValueError, "escaped"):
                runtime._confined_terminal_path(root / "outside.json")
            with self.assertRaisesRegex(ValueError, "authoritative"):
                runtime._confined_terminal_path(runtime.layout.root / "other.json")

    def test_prepare_conflict_never_overwrites_another_runs_terminal_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "shared-run"
            existing = RunConfig.new("noop", run_dir, display_name="existing")
            existing_storage = RunStorage(existing)
            existing_manifest = existing_storage.prepare()
            before_config = existing_storage.layout.config.read_bytes()
            before_result = existing_storage.layout.result.read_bytes()

            conflicting = RunConfig.new("noop", run_dir, display_name="conflicting")
            runtime = WorkerRuntime(conflicting, stream=io.StringIO())

            self.assertEqual(runtime.execute(), 1)
            self.assertEqual(existing_storage.layout.config.read_bytes(), before_config)
            self.assertEqual(existing_storage.layout.result.read_bytes(), before_result)
            self.assertEqual(
                ResultManifest.from_dict(json.loads(before_result)).run_id,
                existing_manifest.run_id,
            )

    def test_result_collection_ignores_matplotlib_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new(
                "noop",
                root / "run",
                output={"history_db": str(root / "history.sqlite3")},
            )
            runtime = WorkerRuntime(config, stream=io.StringIO())
            runtime.manifest = runtime.storage.prepare()
            figure = runtime.layout.gallery / "sample_00000.png"
            figure.write_bytes(b"png")
            cache = runtime.layout.gallery / ".matplotlib-cache" / "fontlist.json"
            cache.parent.mkdir()
            cache.write_text("{}", encoding="utf-8")

            runtime._collect_results()

            paths = [artifact.path for artifact in runtime.manifest.artifacts]
            self.assertIn("gallery/sample_00000.png", paths)
            self.assertFalse(any(".matplotlib-cache" in path for path in paths))

    def test_full_pipeline_respects_disabled_prediction_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "unlabeled dataset"
            dataset.mkdir()
            config = RunConfig.new(
                "full_pipeline",
                root / "run",
                input={"dataset_paths": [str(dataset)]},
                training={"enabled": False},
                prediction={"enabled": False},
                evaluation={"enabled": False},
                output={"history_db": str(root / "history.sqlite3")},
            )
            runtime = WorkerRuntime(config, stream=io.StringIO())
            runtime.run_command = mock.Mock()

            self.assertEqual(runtime.execute(), 0)

            stages = [call.args[0] for call in runtime.run_command.call_args_list]
            self.assertEqual(stages, ["inspect"])

    def test_module_entry_executes_without_gui_or_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = RunConfig.new(
                "noop",
                root / "run",
                output={"history_db": str(root / "history.sqlite3")},
            )
            config_path = root / "input config.json"
            config_path.write_text(config.to_json(), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "spimaging.worker", "--config", str(config_path)],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stdout + result.stderr)
            final = ResultManifest.from_dict(
                json.loads((root / "run" / "result_manifest.json").read_text(encoding="utf-8"))
            )
            self.assertEqual(final.status, "succeeded")

    def test_invalid_config_exits_two_with_concise_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.json"
            path.write_text("{}", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "spimaging.worker", "--config", str(path)],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("schema_version", result.stderr)
            self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_help_documents_config_contract(self) -> None:
        help_text = build_parser().format_help()
        self.assertIn("--config", help_text)
        self.assertIn("schema-v1", help_text)


if __name__ == "__main__":
    unittest.main()
