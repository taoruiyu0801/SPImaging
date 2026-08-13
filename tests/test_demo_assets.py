"""Integrity checks for the small Day 19 demo data and checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "demo_checkpoint" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DemoAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_demo_samples_match_manifest(self) -> None:
        dataset = self.manifest["dataset"]
        dataset_dir = REPOSITORY_ROOT / dataset["path"]
        samples = dataset["samples"]

        self.assertEqual(dataset["sample_count"], len(samples))
        self.assertEqual(dataset["sample_total_bytes"], sum(item["bytes"] for item in samples))
        self.assertEqual(
            sorted(path.name for path in dataset_dir.glob("sample_*.npz")),
            sorted(item["name"] for item in samples),
        )
        for item in samples:
            with self.subTest(sample=item["name"]):
                path = dataset_dir / item["name"]
                self.assertEqual(path.stat().st_size, item["bytes"])
                self.assertEqual(sha256(path), item["sha256"])

    def test_checkpoint_matches_manifest(self) -> None:
        checkpoint = self.manifest["checkpoint"]
        path = REPOSITORY_ROOT / checkpoint["path"]

        self.assertEqual(path.stat().st_size, checkpoint["bytes"])
        self.assertEqual(sha256(path), checkpoint["sha256"])
        self.assertEqual(checkpoint["model_name"], "simple3d")
        self.assertEqual(checkpoint["method_family"], "supervised")

    def test_manifest_uses_portable_paths_and_records_time_limit(self) -> None:
        serialized = json.dumps(self.manifest, ensure_ascii=False)
        self.assertNotIn("C:\\\\Users", serialized)
        self.assertFalse(Path(self.manifest["dataset"]["path"]).is_absolute())
        self.assertFalse(Path(self.manifest["checkpoint"]["path"]).is_absolute())
        verification = self.manifest["verification"]
        self.assertTrue(verification["passed_time_limit"])
        self.assertLess(verification["forced_cpu"]["total_duration_seconds"], 600)


if __name__ == "__main__":
    unittest.main()
