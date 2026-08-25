"""Acceptance tests for deterministic, redistributable public demo assets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import generate_synthetic_demo as generator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = REPOSITORY_ROOT / "public_demo"
MANIFEST_PATH = PUBLIC_ROOT / "manifest.json"
GENERATOR_PATH = REPOSITORY_ROOT / "scripts" / "generate_synthetic_demo.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_proves_analytic_cc0_provenance(manifest: dict[str, object]) -> None:
    provenance = manifest["provenance"]
    assert manifest["schema_version"] == 1
    assert manifest["release"] == "0.2.0-beta.1"
    assert provenance == {
        "asset_license": "CC0-1.0",
        "description": (
            "RGB, metric depth and SPAD counts are constructed from analytic geometry "
            "and fixed pseudo-random seeds in the generator source."
        ),
        "external_image_data": False,
        "external_source_files": [],
        "license_notice": "CC0_NOTICE.md",
        "source_kind": "analytic_procedural_geometry",
    }
    assert manifest["generator"]["path"] == "scripts/generate_synthetic_demo.py"
    assert manifest["generator"]["sha256"] == sha256(GENERATOR_PATH)
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "C:\\Users" not in serialized
    assert "example_data" not in GENERATOR_PATH.read_text(encoding="utf-8")
    assert "demo_checkpoint" not in GENERATOR_PATH.read_text(encoding="utf-8")


def test_four_samples_match_hashes_and_safe_schema(manifest: dict[str, object]) -> None:
    dataset = manifest["dataset"]
    samples = dataset["samples"]
    assert dataset["sample_count"] == 4
    assert len(samples) == 4
    assert sorted(path.name for path in (PUBLIC_ROOT / "dataset").glob("sample_*.npz")) == [
        f"sample_{index:05d}.npz" for index in range(4)
    ]

    required_fields = {
        "counts",
        "depth_m",
        "rgb",
        "albedo",
        "intensity",
        "xhat",
        "x",
        "surface_model",
        "source_mode",
        "scene",
        "sample_id",
        "bin_size",
        "mean_signal_photons",
        "mean_background_photons",
        "sbr",
        "generator_seed",
        "asset_license",
        "generator_version",
    }
    for index, record in enumerate(samples):
        path = PUBLIC_ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert required_fields.issubset(archive.files)
            assert archive["counts"].shape == (1024, 64, 64)
            assert archive["counts"].dtype == np.uint16
            assert archive["depth_m"].shape == (64, 64)
            assert archive["depth_m"].dtype == np.float32
            assert archive["rgb"].shape == (64, 64, 3)
            assert archive["rgb"].dtype == np.uint8
            assert archive["xhat"].shape == (3, 64, 64)
            assert archive["x"].shape == (3, 64, 64)
            assert all(not archive[field].dtype.hasobject for field in archive.files)
            assert np.isfinite(archive["counts"]).all()
            assert np.isfinite(archive["depth_m"]).all()
            assert float(archive["depth_m"].min()) > 0.0
            assert str(archive["source_mode"]) == "synthetic"
            assert str(archive["surface_model"]) == "single"
            assert str(archive["asset_license"]) == "CC0-1.0"
            assert int(archive["sample_id"]) == index
            assert int(archive["generator_seed"]) == record["seed"]


def test_index_and_manifest_hashes_match(manifest: dict[str, object]) -> None:
    index_record = manifest["dataset"]["index"]
    index_path = PUBLIC_ROOT / index_record["path"]
    assert index_path.stat().st_size == index_record["bytes"]
    assert sha256(index_path) == index_record["sha256"]
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert [row["file"] for row in rows] == [f"sample_{index:05d}.npz" for index in range(4)]
    assert [row["sha256"] for row in rows] == [
        record["sha256"] for record in manifest["dataset"]["samples"]
    ]


def test_samples_reproduce_byte_for_byte(tmp_path: Path, manifest: dict[str, object]) -> None:
    reproduced_root = tmp_path / "reproduced public demo"
    generated_manifest = generator.generate(reproduced_root, samples_only=True)
    generator.verify(reproduced_root, require_checkpoint=False)
    reproduced = json.loads(generated_manifest.read_text(encoding="utf-8"))

    expected_records = manifest["dataset"]["samples"]
    actual_records = reproduced["dataset"]["samples"]
    assert [item["sha256"] for item in actual_records] == [
        item["sha256"] for item in expected_records
    ]
    assert [item["bytes"] for item in actual_records] == [item["bytes"] for item in expected_records]
    assert reproduced["dataset"]["index"]["sha256"] == manifest["dataset"]["index"]["sha256"]


def test_checkpoint_is_hash_pinned_safe_and_model_compatible(manifest: dict[str, object]) -> None:
    torch = pytest.importorskip("torch")
    from spimaging.training_common.networks import build_model

    checkpoint = manifest["checkpoint"]
    path = PUBLIC_ROOT / checkpoint["path"]
    assert checkpoint["status"] == "generated"
    assert checkpoint["model_name"] == "simple3d"
    assert checkpoint["method_family"] == "supervised"
    assert path.stat().st_size == checkpoint["bytes"]
    assert sha256(path) == checkpoint["sha256"]

    payload = torch.load(path, map_location="cpu", weights_only=True)
    sample_hashes = [record["sha256"] for record in manifest["dataset"]["samples"]]
    assert payload["training_sample_sha256"] == sample_hashes
    assert checkpoint["recipe"]["training_sample_sha256"] == sample_hashes
    assert payload["asset_license"] == "CC0-1.0"
    assert payload["training_metrics"]["optimizer_steps"] == 32
    assert payload["training_metrics"]["final_kl"] < payload["training_metrics"]["initial_kl"]

    args = payload["args"]
    model = build_model(
        payload["model_name"],
        in_channels=1,
        base_channels=int(args["base_channels"]),
        num_blocks=int(args["num_blocks"]),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    with np.load(PUBLIC_ROOT / manifest["dataset"]["samples"][0]["path"], allow_pickle=False) as sample:
        counts = generator._downsample_time(sample["counts"], int(args["temporal_downsample"]))
    inputs = torch.from_numpy(generator._normalize_counts(counts)[None, None, ...])
    with torch.no_grad():
        output = model(inputs)
    assert tuple(output.shape) == (1, 1, 16, 64, 64)
    assert torch.isfinite(output).all()


def test_checkpoint_recipe_reproduces_in_locked_runtime(
    tmp_path: Path,
    manifest: dict[str, object],
) -> None:
    torch = pytest.importorskip("torch")
    locked_torch = manifest["checkpoint"]["recipe"]["torch_version"]
    if torch.__version__ != locked_torch:
        pytest.skip(f"exact checkpoint bytes require locked torch {locked_torch}")
    reproduced_root = tmp_path / "full reproduction"
    reproduced_manifest_path = generator.generate(reproduced_root)
    generator.verify(reproduced_root)
    reproduced = json.loads(reproduced_manifest_path.read_text(encoding="utf-8"))

    assert reproduced["checkpoint"]["sha256"] == manifest["checkpoint"]["sha256"]
    assert reproduced["checkpoint"]["bytes"] == manifest["checkpoint"]["bytes"]
    assert reproduced["checkpoint"]["recipe"]["final_kl"] == pytest.approx(
        manifest["checkpoint"]["recipe"]["final_kl"],
        rel=0.0,
        abs=0.0,
    )


def test_public_assets_are_not_legacy_demo_files(manifest: dict[str, object]) -> None:
    """Guard against release packaging accidentally copying the old demo assets."""
    legacy_hashes: set[str] = set()
    for legacy_root in (
        REPOSITORY_ROOT / "example_data",
        REPOSITORY_ROOT / "demo_checkpoint",
    ):
        if legacy_root.exists():
            legacy_hashes.update(sha256(path) for path in legacy_root.rglob("*") if path.is_file())

    public_hashes = {record["sha256"] for record in manifest["dataset"]["samples"]}
    public_hashes.add(manifest["checkpoint"]["sha256"])
    assert public_hashes.isdisjoint(legacy_hashes)


def test_license_and_compliance_inventory_present() -> None:
    apache = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (REPOSITORY_ROOT / "NOTICE").read_text(encoding="utf-8")
    third_party = (REPOSITORY_ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    sbom = (REPOSITORY_ROOT / "SBOM.md").read_text(encoding="utf-8")
    cc0_notice = (PUBLIC_ROOT / "CC0_NOTICE.md").read_text(encoding="utf-8")
    cc0_legal = (PUBLIC_ROOT / "CC0-1.0.txt").read_text(encoding="utf-8")

    assert "Apache License" in apache and "Version 2.0, January 2004" in apache
    assert "SPImaging contributors" in notice
    assert "CC0 1.0 Universal" in cc0_notice
    assert "CC0 1.0 Universal" in cc0_legal
    for component in ("NumPy", "PyTorch", "PySide6", "Inno Setup", "conda-pack"):
        assert component in third_party
        assert component in sbom
