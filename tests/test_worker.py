"""Tests for the isolated schema-v1 worker."""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from spimaging.appcore.config import RunConfig
from spimaging.appcore.events import EventType
from spimaging.appcore.storage import ResultManifest
from spimaging.worker import WorkerRuntime, build_parser


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class WorkerRuntimeTests(unittest.TestCase):
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
