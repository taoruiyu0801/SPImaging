"""Tests for the lightweight SPImaging desktop demo interface."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from spimaging import gui


class GuiCommandTests(unittest.TestCase):
    def test_command_uses_current_python_and_preserves_paths_with_spaces(self) -> None:
        command = gui.build_demo_command(
            Path("C:/data set"), Path("C:/demo output"), overwrite=True
        )
        self.assertEqual(command[:4], [sys.executable, "-u", "-m", "spimaging.demo"])
        self.assertEqual(
            command[command.index("--dataset_dir") + 1], str(Path("C:/data set"))
        )
        self.assertEqual(
            command[command.index("--output_dir") + 1], str(Path("C:/demo output"))
        )
        self.assertEqual(command[-1], "--overwrite")

    def test_command_omits_overwrite_by_default(self) -> None:
        command = gui.build_demo_command("data", "output", overwrite=False)
        self.assertNotIn("--overwrite", command)


class GuiValidationTests(unittest.TestCase):
    def test_valid_paths_are_resolved_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "sample_00000.npz").write_bytes(b"sample")
            (dataset / "sample_00001.npz").write_bytes(b"sample")
            output = root / "new output"

            actual_dataset, actual_output = gui.validate_gui_inputs(
                str(dataset), str(output), overwrite=False
            )

            self.assertEqual(actual_dataset, dataset.resolve())
            self.assertEqual(actual_output, output.resolve())
            self.assertFalse(output.exists())

    def test_nonempty_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "one.npz").write_bytes(b"sample")
            (dataset / "two.npz").write_bytes(b"sample")
            output = root / "output"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "覆盖已有演示结果"):
                gui.validate_gui_inputs(str(dataset), str(output), overwrite=False)
            self.assertEqual(
                gui.validate_gui_inputs(str(dataset), str(output), overwrite=True),
                (dataset.resolve(), output.resolve()),
            )

    def test_dataset_requires_two_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "dataset"
            dataset.mkdir()
            (dataset / "one.npz").write_bytes(b"sample")
            with self.assertRaisesRegex(ValueError, "至少需要 2 个"):
                gui.validate_gui_inputs(str(dataset), str(dataset.parent / "output"), False)


class GuiProgressTests(unittest.TestCase):
    def test_stage_start_and_completion_update_progress(self) -> None:
        started = gui.progress_from_line("[2/4] Minimal training...")
        completed = gui.progress_from_line("[2/4] Minimal training... OK (1.2s)")
        self.assertEqual(
            started,
            gui.ProgressUpdate(2, 4, 25, "Minimal training", False),
        )
        self.assertEqual(
            completed,
            gui.ProgressUpdate(2, 4, 50, "Minimal training", True),
        )

    def test_unrelated_or_invalid_lines_are_ignored(self) -> None:
        self.assertIsNone(gui.progress_from_line("epoch 1 loss 0.25"))
        self.assertIsNone(gui.progress_from_line("[5/4] impossible..."))


class GuiSummaryTests(unittest.TestCase):
    def test_load_and_format_success_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            expected = {
                "status": "success",
                "sample_count": 4,
                "total_duration_seconds": 2.5,
                "metrics": {
                    "demo-simple3d": {
                        "mae_m": 0.1,
                        "rmse_m": 0.2,
                        "abs_rel": 0.03,
                    }
                },
            }
            (output / "demo_summary.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )

            summary = gui.load_demo_summary(output)

            self.assertEqual(summary, expected)
            formatted = gui.format_summary(summary)
            self.assertIn("总耗时 2.5 秒", formatted)
            self.assertIn("MAE 0.1 m", formatted)
            self.assertIn("AbsRel 0.03", formatted)

    def test_missing_or_failed_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "无法读取演示摘要"):
                gui.load_demo_summary(output)
            (output / "demo_summary.json").write_text(
                json.dumps({"status": "failed"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "没有成功状态"):
                gui.load_demo_summary(output)


if __name__ == "__main__":
    unittest.main()
