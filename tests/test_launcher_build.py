"""Regression tests for the dependency-free frozen launcher build."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "packaging" / "launcher" / "entrypoint.py"
BUILD_SCRIPT = ROOT / "packaging" / "scripts" / "Build-Launcher.ps1"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("spimaging_launcher_entrypoint", ENTRYPOINT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {ENTRYPOINT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherBuildTests(unittest.TestCase):
    def test_tcl_self_test_passes_in_source_python(self) -> None:
        module = load_entrypoint()
        self.assertEqual(module.tcl_self_test(), 0)

    def test_build_isolates_conda_and_rejects_broken_tcl_bundle(self) -> None:
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        required_contract = (
            "[string]$PythonPath",
            '$_.Name -like "CONDA_*"',
            '"TCL_LIBRARY"',
            '"TK_LIBRARY"',
            '"PYTHONNOUSERSITE"',
            "--launcher-tcl-self-test",
            "Refusing to publish this build",
        )
        for marker in required_contract:
            with self.subTest(marker=marker):
                self.assertIn(marker, script)


if __name__ == "__main__":
    unittest.main()
