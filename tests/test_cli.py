"""Command-line contract tests for every public SPImaging entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SharedNumericTypeTests(unittest.TestCase):
    """Exercise inclusive and exclusive boundaries in the shared CLI types."""

    @classmethod
    def setUpClass(cls) -> None:
        from spimaging import cli

        cls.cli = cli

    def assert_rejected(self, converter, *values: str) -> None:
        for value in values:
            with self.subTest(converter=converter.__name__, value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    converter(value)

    def test_integer_boundaries(self) -> None:
        self.assertEqual(self.cli.positive_int("1"), 1)
        self.assert_rejected(self.cli.positive_int, "0", "-1")

        self.assertEqual(self.cli.nonnegative_int("0"), 0)
        self.assert_rejected(self.cli.nonnegative_int, "-1")

        self.assertEqual(self.cli.random_seed("0"), 0)
        self.assertEqual(self.cli.random_seed("4294967295"), 4294967295)
        self.assert_rejected(self.cli.random_seed, "-1", "4294967296")

        self.assertEqual(self.cli.parameter_index("1"), 1)
        self.assertEqual(self.cli.parameter_index("10"), 10)
        self.assert_rejected(self.cli.parameter_index, "0", "11")

        self.assertEqual(self.cli.positive_odd_int("1"), 1)
        self.assertEqual(self.cli.positive_odd_int("3"), 3)
        self.assert_rejected(self.cli.positive_odd_int, "0", "2")
        self.assertEqual(self.cli.model_base_channels("256"), 256)
        self.assertEqual(self.cli.model_num_blocks("100"), 100)
        self.assertEqual(self.cli.super_resolution_scale("64"), 64)
        self.assert_rejected(self.cli.model_base_channels, "257")
        self.assert_rejected(self.cli.model_num_blocks, "101")
        self.assert_rejected(self.cli.super_resolution_scale, "65")

    def test_float_boundaries(self) -> None:
        self.assertEqual(self.cli.positive_float("0.001"), 0.001)
        self.assert_rejected(self.cli.positive_float, "0", "-0.1", "nan", "inf")

        self.assertEqual(self.cli.nonnegative_float("0"), 0.0)
        self.assert_rejected(self.cli.nonnegative_float, "-0.1", "nan", "inf")

    def test_interval_boundaries(self) -> None:
        self.assertEqual(self.cli.fraction("0"), 0.0)
        self.assertEqual(self.cli.fraction("0.999"), 0.999)
        self.assert_rejected(self.cli.fraction, "-0.001", "1", "nan", "inf")

        self.assertEqual(self.cli.unit_interval("0"), 0.0)
        self.assertEqual(self.cli.unit_interval("1"), 1.0)
        self.assert_rejected(self.cli.unit_interval, "-0.001", "1.001", "nan", "inf")

        self.assertEqual(self.cli.positive_unit_interval("0.001"), 0.001)
        self.assertEqual(self.cli.positive_unit_interval("1"), 1.0)
        self.assert_rejected(
            self.cli.positive_unit_interval,
            "0",
            "-0.001",
            "1.001",
            "nan",
            "inf",
        )


class CliContractTests(unittest.TestCase):
    MODULE_ENTRIES = (
        "spimaging.generation.pipeline",
        "spimaging.supervised_training.train",
        "spimaging.self_supervised_training.train",
        "spimaging.testing.predict",
        "spimaging.testing.evaluate",
        "spimaging.testing.verify",
        "spimaging.testing.browse",
        "spimaging.demo",
    )

    def run_entry(
        self,
        entry: str,
        arguments: list[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        if entry == "main.py":
            command = [sys.executable, str(REPOSITORY_ROOT / "main.py"), *arguments]
        else:
            command = [sys.executable, "-m", entry, *arguments]

        environment = os.environ.copy()
        prior_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(REPOSITORY_ROOT), prior_pythonpath)
            if part
        )
        environment["MPLBACKEND"] = "Agg"
        environment["MPLCONFIGDIR"] = str(cwd / ".matplotlib")

        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )

    def assert_user_error(self, result: subprocess.CompletedProcess[str]) -> None:
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 2, output)
        self.assertIn("error:", output.lower(), output)
        self.assertIn("--help", output, output)
        self.assertNotIn("usage:", output.lower(), output)
        self.assertNotIn("Traceback (most recent call last)", output, output)

    def parse_module_args(self, module_name: str, arguments: list[str]):
        module = importlib.import_module(module_name)
        build_parser = getattr(module, "build_parser", None)
        if build_parser is not None:
            return build_parser().parse_args(arguments)

        with mock.patch.object(sys, "argv", [module_name, *arguments]):
            return module.parse_args()

    def test_all_independent_entries_have_working_help(self) -> None:
        entries = (*self.MODULE_ENTRIES, "main.py")
        with tempfile.TemporaryDirectory() as temporary_directory:
            cwd = Path(temporary_directory)
            for entry in entries:
                with self.subTest(entry=entry):
                    result = self.run_entry(entry, ["--help"], cwd)
                    output = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 0, output)
                    self.assertIn("usage:", result.stdout.lower(), output)
                    self.assertIn("(default:", result.stdout.lower(), output)
                    self.assertNotIn("Traceback (most recent call last)", output, output)

    def test_every_public_option_has_help_text(self) -> None:
        modules = (*self.MODULE_ENTRIES, "main")
        for module_name in modules:
            with self.subTest(module=module_name):
                parser = importlib.import_module(module_name).build_parser()
                for action in parser._actions:
                    if isinstance(action, argparse._HelpAction):
                        continue
                    self.assertNotIn(
                        action.help,
                        (None, argparse.SUPPRESS),
                        f"{module_name}:{action.dest} has no help text",
                    )

    def test_typical_invalid_numbers_exit_with_argparse_status(self) -> None:
        cases = (
            ("generate resolution", "spimaging.generation.pipeline", ["--res", "0"]),
            ("main parameter index", "main.py", ["--param_idx", "11"]),
            (
                "supervised epochs",
                "spimaging.supervised_training.train",
                ["--dataset_dir", "missing-data", "--epochs", "0"],
            ),
            (
                "self-supervised downsample",
                "spimaging.self_supervised_training.train",
                ["--dataset_dir", "missing-data", "--temporal_downsample", "0"],
            ),
            (
                "evaluation figure index",
                "spimaging.testing.evaluate",
                [
                    "--checkpoint",
                    "missing.pt",
                    "--dataset_dir",
                    "missing-data",
                    "--output_dir",
                    "evaluation",
                    "--figure_index",
                    "-1",
                ],
            ),
            (
                "verification index",
                "spimaging.testing.verify",
                ["--dataset_dir", "missing-data", "--index", "-1"],
            ),
            (
                "browser cell size",
                "spimaging.testing.browse",
                ["--dataset_dir", "missing-data", "--cell_size", "0"],
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            cwd = Path(temporary_directory)
            for label, entry, arguments in cases:
                with self.subTest(case=label):
                    self.assert_user_error(self.run_entry(entry, arguments, cwd))

    def test_verify_rejects_conflicting_sample_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_entry(
                "spimaging.testing.verify",
                ["--dataset_dir", "missing-data", "--index", "0", "--random"],
                Path(temporary_directory),
            )
        self.assert_user_error(result)
        self.assertIn("not allowed with argument", result.stderr.lower())

    def test_missing_inputs_do_not_create_outputs_or_show_tracebacks(self) -> None:
        cases = (
            (
                "generate",
                "spimaging.generation.pipeline",
                [
                    "--dataset_mode",
                    "middlebury",
                    "--middlebury_root",
                    "missing-middlebury",
                    "--surface_model",
                    "neighborhood_mix",
                    "--output_dir",
                    "generate-output",
                ],
                "generate-output",
            ),
            (
                "main",
                "main.py",
                [
                    "--dataset_mode",
                    "middlebury",
                    "--middlebury_root",
                    "missing-middlebury",
                    "--surface_model",
                    "neighborhood_mix",
                    "--display",
                    "none",
                    "--output_dir",
                    "main-output",
                ],
                "main-output",
            ),
            (
                "supervised training",
                "spimaging.supervised_training.train",
                ["--dataset_dir", "missing-data", "--output_dir", "train-output"],
                "train-output",
            ),
            (
                "self-supervised training",
                "spimaging.self_supervised_training.train",
                ["--dataset_dir", "missing-data", "--output_dir", "selfsup-output"],
                "selfsup-output",
            ),
            (
                "prediction",
                "spimaging.testing.predict",
                [
                    "--checkpoint",
                    "missing.pt",
                    "--sample_file",
                    "missing.npz",
                    "--output_npz",
                    "predict-output/prediction.npz",
                ],
                "predict-output",
            ),
            (
                "evaluation",
                "spimaging.testing.evaluate",
                [
                    "--checkpoint",
                    "missing.pt",
                    "--dataset_dir",
                    "missing-data",
                    "--output_dir",
                    "evaluate-output",
                ],
                "evaluate-output",
            ),
            (
                "verification",
                "spimaging.testing.verify",
                [
                    "--dataset_dir",
                    "missing-data",
                    "--output_fig",
                    "verify-output/preview.png",
                ],
                "verify-output",
            ),
            (
                "browser",
                "spimaging.testing.browse",
                [
                    "--dataset_dir",
                    "missing-data",
                    "--output_dir",
                    "browse-output",
                ],
                "browse-output",
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            cwd = Path(temporary_directory)
            for label, entry, arguments, output_name in cases:
                with self.subTest(case=label):
                    output_path = cwd / output_name
                    self.assertFalse(output_path.exists())
                    result = self.run_entry(entry, arguments, cwd)
                    self.assert_user_error(result)
                    self.assertFalse(
                        output_path.exists(),
                        f"{label} created {output_path} after rejecting its input",
                    )

    def test_corrupt_inputs_do_not_create_outputs_or_show_tracebacks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cwd = Path(temporary_directory)
            corrupt_dataset = cwd / "corrupt-data"
            corrupt_dataset.mkdir()
            (corrupt_dataset / "sample_00000.npz").write_bytes(b"not an npz archive")
            corrupt_checkpoint = cwd / "corrupt.pt"
            corrupt_checkpoint.write_bytes(b"not a checkpoint")

            middlebury_scene = cwd / "middlebury" / "scene"
            middlebury_scene.mkdir(parents=True)
            (middlebury_scene / "view1.png").write_bytes(b"not a png")
            (middlebury_scene / "disp1.png").write_bytes(b"not a png")

            valid_sample = (
                REPOSITORY_ROOT
                / "example_data"
                / "nyuv2_raw_single_random_snr"
                / "sample_00000.npz"
            )
            cases = (
                (
                    "generate",
                    "spimaging.generation.pipeline",
                    [
                        "--dataset_mode",
                        "middlebury",
                        "--middlebury_root",
                        str(cwd / "middlebury"),
                        "--surface_model",
                        "neighborhood_mix",
                        "--output_dir",
                        "generate-output",
                    ],
                    "generate-output",
                ),
                (
                    "supervised training",
                    "spimaging.supervised_training.train",
                    ["--dataset_dir", str(corrupt_dataset), "--output_dir", "train-output"],
                    "train-output",
                ),
                (
                    "self-supervised training",
                    "spimaging.self_supervised_training.train",
                    ["--dataset_dir", str(corrupt_dataset), "--output_dir", "selfsup-output"],
                    "selfsup-output",
                ),
                (
                    "prediction",
                    "spimaging.testing.predict",
                    [
                        "--checkpoint",
                        str(corrupt_checkpoint),
                        "--sample_file",
                        str(valid_sample),
                        "--output_npz",
                        "predict-output/prediction.npz",
                    ],
                    "predict-output",
                ),
                (
                    "evaluation",
                    "spimaging.testing.evaluate",
                    [
                        "--checkpoint",
                        str(corrupt_checkpoint),
                        "--dataset_dir",
                        str(valid_sample.parent),
                        "--output_dir",
                        "evaluate-output",
                    ],
                    "evaluate-output",
                ),
                (
                    "verification",
                    "spimaging.testing.verify",
                    [
                        "--dataset_dir",
                        str(corrupt_dataset),
                        "--output_fig",
                        "verify-output/preview.png",
                    ],
                    "verify-output",
                ),
                (
                    "browser",
                    "spimaging.testing.browse",
                    [
                        "--dataset_dir",
                        str(corrupt_dataset),
                        "--output_dir",
                        "browse-output",
                    ],
                    "browse-output",
                ),
            )

            for label, entry, arguments, output_name in cases:
                with self.subTest(case=label):
                    output_path = cwd / output_name
                    result = self.run_entry(entry, arguments, cwd)
                    self.assert_user_error(result)
                    self.assertFalse(output_path.exists(), f"{label} created {output_path}")

    def test_canonical_names_and_legacy_aliases_share_destinations(self) -> None:
        cases = (
            (
                "spimaging.testing.predict",
                ["--checkpoint", "model.pt", "--sample_file", "canonical.npz"],
                {"sample_file": "canonical.npz"},
            ),
            (
                "spimaging.testing.predict",
                ["--checkpoint", "model.pt", "--sample", "legacy.npz"],
                {"sample_file": "legacy.npz"},
            ),
            (
                "spimaging.testing.verify",
                [
                    "--dataset_dir",
                    "data",
                    "--sample_name",
                    "canonical.npz",
                    "--output_fig",
                    "canonical.png",
                ],
                {"sample_name": "canonical.npz", "output_fig": "canonical.png"},
            ),
            (
                "spimaging.testing.verify",
                [
                    "--dataset_dir",
                    "data",
                    "--sample",
                    "legacy.npz",
                    "--save_fig",
                    "legacy.png",
                ],
                {"sample_name": "legacy.npz", "output_fig": "legacy.png"},
            ),
            (
                "spimaging.testing.browse",
                ["--dataset_dir", "data", "--output_dir", "canonical-output"],
                {"output_dir": "canonical-output"},
            ),
            (
                "spimaging.testing.browse",
                ["--dataset_dir", "data", "--save_dir", "legacy-output"],
                {"output_dir": "legacy-output"},
            ),
            (
                "main",
                ["--output_fig", "canonical.png"],
                {"output_fig": "canonical.png"},
            ),
            (
                "main",
                ["--save_fig", "legacy.png"],
                {"output_fig": "legacy.png"},
            ),
        )

        for module_name, arguments, expected in cases:
            with self.subTest(module=module_name, arguments=arguments):
                parsed = self.parse_module_args(module_name, arguments)
                for destination, value in expected.items():
                    self.assertEqual(getattr(parsed, destination), value)

    def test_overwrite_flag_is_available_on_every_output_entry(self) -> None:
        cases = (
            ("spimaging.generation.pipeline", ["--overwrite"]),
            (
                "spimaging.supervised_training.train",
                ["--dataset_dir", "data", "--overwrite"],
            ),
            (
                "spimaging.self_supervised_training.train",
                ["--dataset_dir", "data", "--overwrite"],
            ),
            (
                "spimaging.testing.predict",
                ["--checkpoint", "model.pt", "--sample_file", "sample.npz", "--overwrite"],
            ),
            (
                "spimaging.testing.evaluate",
                [
                    "--checkpoint",
                    "model.pt",
                    "--dataset_dir",
                    "data",
                    "--output_dir",
                    "output",
                    "--overwrite",
                ],
            ),
            ("spimaging.testing.verify", ["--dataset_dir", "data", "--overwrite"]),
            ("spimaging.testing.browse", ["--dataset_dir", "data", "--overwrite"]),
            ("main", ["--overwrite"]),
        )

        for module_name, arguments in cases:
            with self.subTest(module=module_name):
                parsed = self.parse_module_args(module_name, arguments)
                self.assertTrue(parsed.overwrite)

    def test_generation_publish_removes_only_stale_owned_outputs(self) -> None:
        from spimaging.generation.pipeline import build_parser, publish_generated_output
        from spimaging.generation.recovery import partial_directory_for

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            staging_dir = partial_directory_for(output_dir)
            output_dir.mkdir()
            staging_dir.mkdir()
            (output_dir / "sample_00000.npz").write_bytes(b"old zero")
            (output_dir / "sample_00001.npz").write_bytes(b"stale one")
            (output_dir / "index.csv").write_text("old index", encoding="utf-8")
            (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
            (staging_dir / "sample_00000.npz").write_bytes(b"new zero")
            (staging_dir / "index.csv").write_text("new index", encoding="utf-8")
            (staging_dir / "generation_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "sample_count": 1,
                        "completed": [
                            {
                                "file": "sample_00000.npz",
                                "bytes": len(b"new zero"),
                                "sha256": hashlib.sha256(b"new zero").hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            publish_generated_output(build_parser(), staging_dir, output_dir, overwrite=True)

            self.assertEqual((output_dir / "sample_00000.npz").read_bytes(), b"new zero")
            self.assertFalse((output_dir / "sample_00001.npz").exists())
            self.assertEqual((output_dir / "index.csv").read_text(encoding="utf-8"), "new index")
            self.assertEqual((output_dir / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_main_propagates_overwrite_to_browse(self) -> None:
        import main as quickstart

        args = argparse.Namespace(display="browse", output_dir="generated", overwrite=True)
        captured = []

        def capture_argv():
            captured.extend(sys.argv)

        with mock.patch("spimaging.testing.browse.main", side_effect=capture_argv):
            quickstart.show_generated_data(args)
        self.assertIn("--overwrite", captured)

    def test_generated_parameter_tables_are_current(self) -> None:
        generator_path = REPOSITORY_ROOT / "scripts" / "generate_cli_reference.py"
        spec = importlib.util.spec_from_file_location("generate_cli_reference", generator_path)
        generator = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(generator)
        data = list(generator.rows())

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            generated_markdown = temporary_root / "参数说明表.md"
            generated_csv = temporary_root / "参数说明表.csv"
            generator.write_markdown(generated_markdown, data)
            generator.write_csv(generated_csv, data)

            checked_in_root = REPOSITORY_ROOT / "record_of_SPI" / "Day_13-14"
            self.assertEqual(
                generated_markdown.read_bytes(),
                (checked_in_root / "参数说明表.md").read_bytes(),
                "参数说明表.md is stale; run: python scripts/generate_cli_reference.py",
            )
            self.assertEqual(
                generated_csv.read_bytes(),
                (checked_in_root / "参数说明表.csv").read_bytes(),
                "参数说明表.csv is stale; run: python scripts/generate_cli_reference.py",
            )


if __name__ == "__main__":
    unittest.main()
