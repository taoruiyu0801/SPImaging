"""Contract tests for the fail-closed public release manifest."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import unittest

from launcher.errors import ManifestError
from launcher.manifest import NvidiaCapability, ReleaseManifest, compare_semver, select_runtime_asset


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_APP_BUILDER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "packaging" / "scripts" / "build_app_asset.py")
)


def valid_manifest_dict(*, cuda: bool = True) -> dict[str, object]:
    def asset(component: str, variant: str, digest: str) -> dict[str, object]:
        asset_id = f"spimaging-{component}-{variant}-0.2.0-beta.1"
        result: dict[str, object] = {
            "asset_id": asset_id,
            "component": component,
            "variant": variant,
            "platform": "windows-x86_64",
            "version": "0.2.0-beta.1",
            "archive_name": f"{asset_id}.zip",
            "archive_format": "zip",
            "archive_size": 12,
            "archive_sha256": digest,
            "unpacked_size": 128,
            "parts": [
                {
                    "name": f"{asset_id}.zip.001",
                    "url": f"https://example.invalid/{asset_id}.zip.001",
                    "size": 12,
                    "sha256": digest,
                }
            ],
            "required_paths": ["python.exe"] if component == "runtime" else ["spimaging/__init__.py"],
            "relocation": None,
            "health_check": None,
            "signature": {"kind": "none", "required": False},
        }
        if variant == "cuda":
            result["min_nvidia_driver"] = "550.0"
        return result

    assets = [asset("runtime", "cpu", _HASH_A), asset("app", "universal", _HASH_B)]
    if cuda:
        assets.append(asset("runtime", "cuda", _HASH_C))
    return {
        "schema_version": 1,
        "product": "SPImaging",
        "release_version": "0.2.0-beta.1",
        "channel": "beta",
        "published_at": "2026-08-23T00:00:00Z",
        "unsigned_beta": True,
        "assets": assets,
        "launch": {
            "runtime_executable": "pythonw.exe",
            "console_executable": "python.exe",
            "app_module": "spimaging.desktop",
            "arguments": [],
        },
    }


class ReleaseManifestTests(unittest.TestCase):
    def test_valid_manifest_round_trip_and_components(self) -> None:
        source = valid_manifest_dict()
        manifest = ReleaseManifest.from_json(json.dumps(source))
        self.assertEqual(manifest.release_version, "0.2.0-beta.1")
        self.assertEqual(manifest.asset("runtime", "cpu").archive_size, 12)
        self.assertEqual(manifest.asset("app", "universal").variant, "universal")

    def test_rejects_missing_hash_insecure_url_and_wrong_part_sum(self) -> None:
        cases = []
        missing_hash = valid_manifest_dict()
        del missing_hash["assets"][0]["parts"][0]["sha256"]  # type: ignore[index]
        cases.append(missing_hash)
        insecure = valid_manifest_dict()
        insecure["assets"][0]["parts"][0]["url"] = "http://example.invalid/a"  # type: ignore[index]
        cases.append(insecure)
        wrong_size = valid_manifest_dict()
        wrong_size["assets"][0]["archive_size"] = 13  # type: ignore[index]
        cases.append(wrong_size)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ManifestError):
                ReleaseManifest.from_dict(case)

    def test_rejects_traversal_and_duplicate_part_names(self) -> None:
        traversal = valid_manifest_dict()
        traversal["assets"][0]["parts"][0]["name"] = "../runtime.zip.001"  # type: ignore[index]
        with self.assertRaisesRegex(ManifestError, "safe relative"):
            ReleaseManifest.from_dict(traversal)
        duplicate = valid_manifest_dict()
        first = duplicate["assets"][0]  # type: ignore[index]
        first["archive_size"] = 24
        first["parts"].append(deepcopy(first["parts"][0]))
        with self.assertRaisesRegex(ManifestError, "duplicate"):
            ReleaseManifest.from_dict(duplicate)

    def test_stable_release_fails_closed_without_required_signatures(self) -> None:
        source = valid_manifest_dict()
        source["channel"] = "stable"
        source["unsigned_beta"] = False
        with self.assertRaisesRegex(ManifestError, "must require a signature"):
            ReleaseManifest.from_dict(source)

    def test_requires_cpu_and_app_but_cuda_is_optional(self) -> None:
        self.assertIsInstance(ReleaseManifest.from_dict(valid_manifest_dict(cuda=False)), ReleaseManifest)
        source = valid_manifest_dict()
        source["assets"] = [item for item in source["assets"] if item["variant"] != "cpu"]  # type: ignore[index]
        with self.assertRaisesRegex(ManifestError, "requires CPU runtime"):
            ReleaseManifest.from_dict(source)

    def test_runtime_layer_version_is_independent_but_app_tracks_release(self) -> None:
        source = valid_manifest_dict(cuda=False)
        runtime = source["assets"][0]  # type: ignore[index]
        runtime["version"] = "0.2.0-runtime.1"
        runtime["asset_id"] = "spimaging-runtime-cpu-0.2.0-runtime.1"
        manifest = ReleaseManifest.from_dict(source)
        self.assertEqual(manifest.asset("runtime", "cpu").version, "0.2.0-runtime.1")

        app_mismatch = deepcopy(source)
        app_mismatch["assets"][1]["version"] = "0.2.0-beta.2"  # type: ignore[index]
        with self.assertRaisesRegex(ManifestError, "app asset.*release_version"):
            ReleaseManifest.from_dict(app_mismatch)

    def test_versions_use_strict_semver(self) -> None:
        for invalid in ("01.2.3", "1.02.3", "1.2", "1.2.3-01", "1.2.3-"):
            source = valid_manifest_dict(cuda=False)
            source["release_version"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ManifestError):
                ReleaseManifest.from_dict(source)

    def test_semver_precedence_supports_prerelease_and_ignores_build_metadata(self) -> None:
        self.assertLess(compare_semver("0.2.0-beta.1", "0.2.0-beta.2"), 0)
        self.assertLess(compare_semver("0.2.0-beta.2", "0.2.0"), 0)
        self.assertEqual(compare_semver("0.2.0+build.1", "0.2.0+build.2"), 0)


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = ReleaseManifest.from_dict(valid_manifest_dict())

    def test_auto_prefers_compatible_cuda(self) -> None:
        selection = select_runtime_asset(self.manifest, "auto", NvidiaCapability(True, "551.61"))
        self.assertEqual(selection.selected_variant, "cuda")
        self.assertFalse(selection.fallback)

    def test_old_driver_and_missing_gpu_fall_back_with_reason(self) -> None:
        old = select_runtime_asset(self.manifest, "cuda", NvidiaCapability(True, "549.99"))
        missing = select_runtime_asset(self.manifest, "auto", NvidiaCapability(False, reason="not found"))
        self.assertEqual(old.selected_variant, "cpu")
        self.assertTrue(old.fallback)
        self.assertIn("低于要求", old.reason)
        self.assertEqual(missing.selected_variant, "cpu")
        self.assertIn("not found", missing.reason)

    def test_explicit_cpu_never_probes_into_cuda(self) -> None:
        selection = select_runtime_asset(self.manifest, "cpu", NvidiaCapability(True, "999.0"))
        self.assertEqual(selection.selected_variant, "cpu")
        self.assertFalse(selection.fallback)


class PublicAppAssetPolicyTests(unittest.TestCase):
    def test_compliance_files_are_mandatory_and_private_roots_are_rejected(self) -> None:
        required = _APP_BUILDER["REQUIRED_FILES"]
        validate_name = _APP_BUILDER["validate_name"]
        self.assertEqual(
            required,
            {"LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md", "SBOM.md"},
        )
        for name in required:
            with self.subTest(name=name):
                self.assertEqual(str(validate_name(name)), name)
        for name in (
            "example_data/private.npz",
            "demo_checkpoint/legacy.pt",
            "record_of_SPI/notes.txt",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "forbidden|allowlist"):
                validate_name(name)


class RuntimePackagingPolicyTests(unittest.TestCase):
    def test_cpu_and_cuda_inputs_include_gui_single_surface_dependencies(self) -> None:
        runtime_root = Path(__file__).resolve().parents[1] / "packaging" / "runtime"
        for variant in ("cpu", "cuda"):
            source = (runtime_root / f"environment-{variant}.in.yml").read_text(encoding="utf-8")
            with self.subTest(variant=variant):
                self.assertIn("pyside6=6.9", source)
                self.assertIn("pytorch=2.5.1", source)
                self.assertIn("torchvision=0.20.1", source)
                self.assertIn("deepinv==0.4.1", source)

    def test_generated_runtime_health_imports_gui_and_deepinv(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "packaging"
            / "scripts"
            / "build_release_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertIn("import PySide6,deepinv", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn('"--runtime-version"', source)

    def test_release_workflow_uses_env_validated_versions_and_manifest_signature(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "windows-release-candidate.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("$semver =", workflow)
        self.assertIn("-Version $env:RELEASE_VERSION", workflow)
        self.assertIn("-RuntimeVersion $env:RUNTIME_VERSION", workflow)
        self.assertNotIn('-Version "${{ inputs.version }}"', workflow)
        self.assertIn("spimaging-release-manifest.json.p7s", workflow)
        self.assertIn("choco install innosetup", workflow)
        self.assertIn("Refusing nested packaging/out/out", workflow)
        self.assertIn("publish_unsigned_beta", workflow)
        self.assertIn("windows-unsigned-beta-approval", workflow)
        self.assertIn("Unsigned beta must never publish the stable installer filename", workflow)
        self.assertIn('$channelTag = "windows-beta"', workflow)

    def test_launcher_combined_health_uses_selected_runtime_and_desktop_smoke(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "launcher" / "bootstrap.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"-m", manifest.launch.app_module, "--smoke-test"', source)
        self.assertIn("activate_many(requests, final_health_check=combined)", source)

    def test_installer_includes_public_compliance_and_cc0_legal_texts(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "packaging" / "inno" / "SPImaging.iss"
        ).read_text(encoding="utf-8")
        for required in (
            "THIRD_PARTY_LICENSES.md",
            "SBOM.md",
            "public_demo\\CC0_NOTICE.md",
            "public_demo\\CC0-1.0.txt",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        build_script = (
            Path(__file__).resolve().parents[1]
            / "packaging"
            / "scripts"
            / "Build-Installer.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("Missing installer legal/compliance input", build_script)
        for required in ("THIRD_PARTY_LICENSES.md", "SBOM.md", "CC0_NOTICE.md", "CC0-1.0.txt"):
            self.assertIn(required, build_script)


if __name__ == "__main__":
    unittest.main()
