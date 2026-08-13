"""Tests for the stable SPImaging demonstration workflow."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from spimaging import demo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_sample(path: Path) -> None:
    counts = np.ones((4, 2, 2), dtype=np.float32)
    depth = np.ones((2, 2), dtype=np.float32)
    np.savez_compressed(path, counts=counts, depth_m=depth)


def write_fake_artifacts(staging_dir: Path, sample: Path, sample_count: int) -> None:
    (staging_dir / "verify").mkdir(parents=True, exist_ok=True)
    (staging_dir / "verify" / f"{sample.stem}.png").write_bytes(b"png")
    (staging_dir / "train").mkdir(parents=True, exist_ok=True)
    (staging_dir / "train" / "last.pt").write_bytes(b"checkpoint")
    (staging_dir / "predict").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        staging_dir / "predict" / "prediction.npz",
        pred_depth_m=np.ones((2, 2), dtype=np.float32),
    )
    (staging_dir / "predict" / "comparison.png").write_bytes(b"png")
    (staging_dir / "evaluate").mkdir(parents=True, exist_ok=True)
    (staging_dir / "evaluate" / "metrics_per_sample.csv").write_text(
        "model,sample,mae_m,rmse_m,abs_rel\n",
        encoding="utf-8",
    )
    (staging_dir / "evaluate" / "metrics_summary.json").write_text(
        json.dumps(
            {
                "demo-simple3d": {
                    "n_samples": sample_count,
                    "mae_m": 0.1,
                    "rmse_m": 0.2,
                    "abs_rel": 0.01,
                }
            }
        ),
        encoding="utf-8",
    )
    (staging_dir / "evaluate" / "comparison.png").write_bytes(b"png")


class DemoParserTests(unittest.TestCase):
    def test_help_uses_stable_public_defaults(self) -> None:
        help_text = demo.build_parser().format_help()
        self.assertIn("spad-demo", help_text)
        self.assertIn(str(demo.DEFAULT_DATASET_DIR), help_text)
        self.assertIn(str(demo.DEFAULT_OUTPUT_DIR), help_text)
        self.assertIn("--overwrite", help_text)

    def test_stage_commands_use_modules_in_required_order(self) -> None:
        root = Path("C:/demo-test")
        stages = demo.build_stages(root / "data", root / "staging", root / "data/sample_00000.npz")
        self.assertEqual([stage.name for stage in stages], ["verify", "train", "predict", "evaluate"])
        self.assertEqual(
            [stage.module for stage in stages],
            [
                "spimaging.testing.verify",
                "spimaging.supervised_training.train",
                "spimaging.testing.predict",
                "spimaging.testing.evaluate",
            ],
        )
        train_args = stages[1].arguments
        self.assertEqual(train_args[train_args.index("--epochs") + 1], "1")
        self.assertEqual(train_args[train_args.index("--max_samples") + 1], "2")
        self.assertEqual(train_args[train_args.index("--temporal_downsample") + 1], "64")

    def test_child_environment_is_headless_without_invalid_warning_filter_syntax(self) -> None:
        environment = demo.child_environment(Path("C:/runtime-cache"))
        self.assertEqual(environment["MPLBACKEND"], "Agg")
        self.assertNotIn(",", environment["PYTHONWARNINGS"])


class DemoWorkflowTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        dataset_dir = root / "dataset with spaces"
        dataset_dir.mkdir()
        write_sample(dataset_dir / "sample_00000.npz")
        write_sample(dataset_dir / "sample_00001.npz")
        return dataset_dir, root / "output with spaces"

    def run_with_fake_stages(
        self,
        dataset_dir: Path,
        output_dir: Path,
        *,
        overwrite: bool = False,
        fail_at: str | None = None,
    ) -> list[str]:
        calls = []

        def runner(stage, **kwargs):
            calls.append(stage.name)
            if stage.name == fail_at:
                raise demo.DemoStageError(stage, 7)
            if stage.name == "evaluate":
                write_fake_artifacts(kwargs["staging_dir"], dataset_dir / "sample_00000.npz", 2)
            return 0.01

        parser = demo.build_parser()
        args = argparse.Namespace(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            overwrite=overwrite,
        )
        with mock.patch("spimaging.demo.importlib.import_module", return_value=object()):
            demo.run_demo(args, parser, stage_runner=runner)
        return calls

    def test_four_stages_publish_validated_outputs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir, output_dir = self.make_fixture(root)
            calls = self.run_with_fake_stages(dataset_dir, output_dir)

            self.assertEqual(calls, ["verify", "train", "predict", "evaluate"])
            summary = json.loads((output_dir / "demo_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual([stage["name"] for stage in summary["stages"]], calls)
            self.assertTrue((output_dir / "train" / "last.pt").is_file())
            self.assertFalse(any(root.glob(".output with spaces.staging-*")))
            serialized = json.dumps(summary, ensure_ascii=False)
            self.assertNotIn(str(root), serialized)

    def test_failed_stage_stops_following_stages_and_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir, output_dir = self.make_fixture(root)

            with self.assertRaises(demo.DemoStageError):
                self.run_with_fake_stages(dataset_dir, output_dir, fail_at="train")

            self.assertFalse(output_dir.exists())
            self.assertFalse(any(root.glob(".output with spaces.staging-*")))

    def test_overwrite_replaces_owned_outputs_and_preserves_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir, output_dir = self.make_fixture(root)
            output_dir.mkdir()
            (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
            (output_dir / "train").mkdir()
            (output_dir / "train" / "stale.pt").write_bytes(b"stale")

            self.run_with_fake_stages(dataset_dir, output_dir, overwrite=True)

            self.assertEqual((output_dir / "keep.txt").read_text(encoding="utf-8"), "keep")
            self.assertFalse((output_dir / "train" / "stale.pt").exists())
            self.assertTrue((output_dir / "train" / "last.pt").is_file())

    def test_invalid_dataset_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            write_sample(dataset_dir / "sample_00000.npz")
            output_dir = root / "output"
            parser = demo.build_parser()
            args = argparse.Namespace(
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                overwrite=False,
            )
            with self.assertRaises(SystemExit) as error:
                demo.prepare_inputs(parser, args)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output_dir.exists())

    def test_run_stage_uses_current_python_module_and_no_shell(self) -> None:
        stage = demo.StageSpec("verify", "checking sample", "example.module", ("--value", "a b"))
        process = mock.MagicMock()
        process.stdout = io.StringIO("done\n")
        process.wait.return_value = 0
        with mock.patch("spimaging.demo.subprocess.Popen", return_value=process) as popen:
            duration = demo.run_stage(
                stage,
                log_file=io.StringIO(),
                environment={},
                dataset_dir=REPOSITORY_ROOT / "data",
                staging_dir=REPOSITORY_ROOT / "staging",
            )
        self.assertGreaterEqual(duration, 0)
        command = popen.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "example.module"])
        self.assertEqual(command[-1], "a b")
        self.assertNotIn("shell", popen.call_args.kwargs)


class DemoInstalledEntryTests(unittest.TestCase):
    def test_module_help_is_headless_and_has_no_traceback(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "spimaging.demo", "--help"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("usage: spad-demo", result.stdout)
        self.assertNotIn("Traceback", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
