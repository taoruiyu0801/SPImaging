"""Offline tests for launcher download, extraction, activation, and updates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from launcher.activation import ActivationManager
from launcher.app import (
    DEFAULT_BETA_MANIFEST_URL,
    _is_update,
    _updates_disabled,
    build_parser as build_launcher_parser,
    load_desktop_preferences,
    runtime_preference,
)
from launcher.archive import safe_extract_zip
from launcher.bootstrap import ProvisionResult, Provisioner
from launcher.download import (
    DownloadResponse,
    LocalDirectoryTransport,
    download_asset,
    download_part,
    sha256_file,
)
from launcher.errors import DownloadError, ExtractionError, HealthCheckError, LauncherError, SignatureError
from launcher.locking import InterProcessLock
from launcher.manifest import (
    AssetPart,
    HealthCheck,
    LaunchSpec,
    NvidiaCapability,
    ReleaseAsset,
    ReleaseManifest,
    SignatureRequirement,
    manifest_to_dict,
)
from launcher.signing import ManifestTrustPolicy
from launcher.update import ManifestCache


class MemoryTransport:
    def __init__(self, content: dict[str, bytes]) -> None:
        self.content = content
        self.starts: list[tuple[str, int]] = []

    def open(self, url: str, start: int = 0) -> DownloadResponse:
        self.starts.append((url, start))
        data = self.content[url]
        status = 206 if start else 200
        headers = {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"} if start else {}
        return DownloadResponse(io.BytesIO(data[start:]), status, headers)


def make_zip(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return stream.getvalue()


def noop_desktop_smoke(_runtime: Path, _app: Path, _manifest: ReleaseManifest) -> object:
    return object()


def asset_for_bytes(
    content: bytes,
    *,
    asset_id: str = "runtime-cpu-test",
    component: str = "runtime",
    variant: str = "cpu",
    required_path: str = "python.exe",
    url: str = "https://example.invalid/asset.001",
    signature: SignatureRequirement | None = None,
) -> ReleaseAsset:
    digest = hashlib.sha256(content).hexdigest()
    return ReleaseAsset(
        asset_id=asset_id,
        component=component,
        variant=variant,
        platform="windows-x86_64",
        version="0.2.0-beta.1",
        archive_name=f"{asset_id}.zip",
        archive_format="zip",
        archive_size=len(content),
        archive_sha256=digest,
        unpacked_size=1024 * 1024,
        parts=(AssetPart(f"{asset_id}.001", url, len(content), digest),),
        required_paths=(required_path,),
        relocation=None,
        health_check=None,
        signature=signature or SignatureRequirement("none", False),
    )


class DownloadTests(unittest.TestCase):
    def test_local_directory_transport_supports_verified_offline_resume(self) -> None:
        content = b"offline-runtime" * 128
        part = AssetPart(
            "runtime.001",
            "https://example.invalid/releases/runtime.001",
            len(content),
            hashlib.sha256(content).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / part.name).write_bytes(content)
            destination = root / "cache" / part.name
            destination.parent.mkdir()
            destination.with_name(destination.name + ".part").write_bytes(content[:31])

            result = download_part(part, destination, LocalDirectoryTransport(root), chunk_size=17)

            self.assertEqual(result.read_bytes(), content)

    def test_resumes_partial_download_and_validates_hash(self) -> None:
        content = b"verified release bytes" * 100
        digest = hashlib.sha256(content).hexdigest()
        part = AssetPart("runtime.001", "https://example.invalid/runtime.001", len(content), digest)
        transport = MemoryTransport({part.url: content})
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / part.name
            partial = destination.with_name(destination.name + ".part")
            partial.write_bytes(content[:137])
            result = download_part(part, destination, transport, chunk_size=31)
            self.assertEqual(result.read_bytes(), content)
            self.assertEqual(transport.starts, [(part.url, 137)])
            self.assertFalse(partial.exists())

    def test_server_ignoring_range_restarts_instead_of_duplicating(self) -> None:
        content = b"abcdef"
        digest = hashlib.sha256(content).hexdigest()
        part = AssetPart("part", "https://example.invalid/part", len(content), digest)

        class IgnoreRange(MemoryTransport):
            def open(self, url: str, start: int = 0) -> DownloadResponse:
                self.starts.append((url, start))
                return DownloadResponse(io.BytesIO(self.content[url]), 200, {})

        transport = IgnoreRange({part.url: content})
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "part"
            destination.with_name("part.part").write_bytes(b"abc")
            download_part(part, destination, transport)
            self.assertEqual(destination.read_bytes(), content)

    def test_truncated_response_preserves_partial_for_next_resume(self) -> None:
        content = b"0123456789"
        digest = hashlib.sha256(content).hexdigest()
        part = AssetPart("part", "https://example.invalid/truncated", len(content), digest)
        transport = MemoryTransport({part.url: content[:4]})
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "part"
            with self.assertRaisesRegex(DownloadError, "incomplete"):
                download_part(part, destination, transport)
            self.assertEqual(destination.with_name("part.part").read_bytes(), content[:4])

    def test_hash_mismatch_fails_and_removes_untrusted_partial(self) -> None:
        part = AssetPart("bad", "https://example.invalid/bad", 3, "0" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "bad"
            with self.assertRaisesRegex(DownloadError, "mismatch"):
                download_part(part, destination, MemoryTransport({part.url: b"bad"}))
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("bad.part").exists())

    def test_required_signature_fails_closed_without_verifier(self) -> None:
        archive = make_zip({"python.exe": b"placeholder"})
        requirement = SignatureRequirement("authenticode", True, signer_thumbprint="A" * 40)
        asset = asset_for_bytes(archive, signature=requirement)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SignatureError):
                download_asset(asset, Path(temporary), MemoryTransport({asset.parts[0].url: archive}))

    def test_invalid_complete_partial_restarts_at_zero_instead_of_looping_on_416(self) -> None:
        content = b"correct content"
        part = AssetPart(
            "runtime.001",
            "https://example.invalid/complete-partial",
            len(content),
            hashlib.sha256(content).hexdigest(),
        )
        transport = MemoryTransport({part.url: content})
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / part.name
            destination.with_name(part.name + ".part").write_bytes(b"x" * len(content))
            self.assertEqual(download_part(part, destination, transport).read_bytes(), content)
            self.assertEqual(transport.starts, [(part.url, 0)])

    def test_http_416_discards_stale_partial_and_retries_once_from_zero(self) -> None:
        content = b"0123456789"
        part = AssetPart(
            "runtime.001",
            "https://example.invalid/range-416",
            len(content),
            hashlib.sha256(content).hexdigest(),
        )

        class Range416ThenSuccess(MemoryTransport):
            def open(self, url: str, start: int = 0) -> DownloadResponse:
                self.starts.append((url, start))
                if start:
                    return DownloadResponse(io.BytesIO(), 416, {})
                return DownloadResponse(io.BytesIO(self.content[url]), 200, {})

        transport = Range416ThenSuccess({part.url: content})
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / part.name
            destination.with_name(part.name + ".part").write_bytes(content[:4])
            self.assertEqual(download_part(part, destination, transport).read_bytes(), content)
            self.assertEqual(transport.starts, [(part.url, 4), (part.url, 0)])


class ExtractionTests(unittest.TestCase):
    def test_extracts_regular_files(self) -> None:
        content = make_zip({"python.exe": b"python", "Lib/module.py": b"ok = True\n"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "runtime.zip"
            archive.write_bytes(content)
            destination = root / "staging"
            extracted = safe_extract_zip(archive, destination, max_unpacked_size=1024)
            self.assertEqual(extracted, len(b"python") + len(b"ok = True\n"))
            self.assertEqual((destination / "Lib" / "module.py").read_text(), "ok = True\n")

    def test_rejects_path_traversal_before_extracting_anything(self) -> None:
        content = make_zip({"safe.txt": b"safe", "../outside.txt": b"owned"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            archive.write_bytes(content)
            destination = root / "staging"
            with self.assertRaisesRegex(ExtractionError, "unsafe"):
                safe_extract_zip(archive, destination, max_unpacked_size=1024)
            self.assertFalse((root / "outside.txt").exists())
            self.assertFalse((destination / "safe.txt").exists())

    def test_rejects_manifest_unpacked_size_overflow(self) -> None:
        content = make_zip({"large.bin": b"x" * 200})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "large.zip"
            archive.write_bytes(content)
            with self.assertRaisesRegex(ExtractionError, "unpacked_size"):
                safe_extract_zip(archive, root / "staging", max_unpacked_size=100)

    def test_rejects_windows_device_names(self) -> None:
        for name in ("NUL.txt", "CONIN$", "CONOUT$.log", "COM¹.txt", "LPT³"):
            content = make_zip({f"folder/{name}": b"unsafe"})
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "device.zip"
                archive.write_bytes(content)
                with self.assertRaisesRegex(ExtractionError, "Windows"):
                    safe_extract_zip(archive, root / "staging", max_unpacked_size=1024)


class ActivationTests(unittest.TestCase):
    def test_activation_and_rollback_are_state_driven(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = ActivationManager(Path(temporary) / "install")
            first = manager.create_staging("runtime")
            (first / "ready").write_text("v1")
            first_path = manager.activate("runtime", "v1-cpu", first, health_check=lambda root: (root / "ready").read_text())
            second = manager.create_staging("runtime")
            (second / "ready").write_text("v2")
            manager.activate("runtime", "v2-cpu", second, health_check=lambda root: (root / "ready").read_text())
            self.assertEqual((manager.active_path("runtime") / "ready").read_text(), "v2")  # type: ignore[operator]
            rolled_back = manager.rollback("runtime", health_check=lambda root: (root / "ready").read_text())
            self.assertEqual(rolled_back, first_path)
            self.assertEqual(manager.active_record("runtime")["previous"], "v2-cpu")  # type: ignore[index]

    def test_failed_health_check_never_changes_active_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = ActivationManager(Path(temporary) / "install")
            first = manager.create_staging("app")
            (first / "ok").write_text("ok")
            manager.activate("app", "v1", first, health_check=lambda _: True)
            broken = manager.create_staging("app")
            with self.assertRaisesRegex(HealthCheckError, "broken"):
                manager.activate(
                    "app",
                    "v2",
                    broken,
                    health_check=lambda _: (_ for _ in ()).throw(HealthCheckError("broken")),
                )
            self.assertEqual(manager.active_record("app")["active"], "v1")  # type: ignore[index]

    def test_repair_replaces_same_release_after_new_stage_passes_health(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = ActivationManager(Path(temporary) / "install")
            first = manager.create_staging("runtime")
            (first / "ready").write_text("old")
            manager.activate("runtime", "v1-cpu", first, health_check=lambda _: True)
            replacement = manager.create_staging("runtime")
            (replacement / "ready").write_text("repaired")
            manager.activate(
                "runtime",
                "v1-cpu",
                replacement,
                health_check=lambda root: (root / "ready").read_text(),
                replace_existing=True,
            )
            self.assertEqual((manager.active_path("runtime") / "ready").read_text(), "repaired")  # type: ignore[operator]


class ProvisioningTests(unittest.TestCase):
    def test_local_archives_provision_runtime_and_app(self) -> None:
        runtime_zip = make_zip({"python.exe": b"private runtime"})
        app_zip = make_zip({"spimaging/__init__.py": b"# app\n"})
        runtime = asset_for_bytes(runtime_zip, url="https://example.invalid/runtime")
        app = asset_for_bytes(
            app_zip,
            asset_id="app-universal-test",
            component="app",
            variant="universal",
            required_path="spimaging/__init__.py",
            url="https://example.invalid/app",
        )
        manifest = ReleaseManifest(
            schema_version=1,
            product="SPImaging",
            release_version="0.2.0-beta.1",
            channel="beta",
            published_at="2026-08-23T00:00:00Z",
            unsigned_beta=True,
            assets=(runtime, app),
            launch=LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ()),
        )
        transport = MemoryTransport({runtime.parts[0].url: runtime_zip, app.parts[0].url: app_zip})
        with tempfile.TemporaryDirectory() as temporary:
            result = Provisioner(
                Path(temporary) / "install", transport, desktop_smoke=noop_desktop_smoke
            ).provision(
                manifest, "cpu", NvidiaCapability(False)
            )
            self.assertEqual((result.runtime_path / "python.exe").read_bytes(), b"private runtime")
            self.assertTrue((result.app_path / "spimaging" / "__init__.py").is_file())
            self.assertEqual(result.runtime_variant, "cpu")

    def test_runtime_relocation_runs_at_final_path_before_health_check(self) -> None:
        runtime_zip = make_zip(
            {
                "python.exe": b"python",
                "Scripts/conda-unpack.exe": b"relocator",
            }
        )
        app_zip = make_zip({"spimaging/__init__.py": b"# app\n"})
        runtime = replace(
            asset_for_bytes(runtime_zip, url="https://example.invalid/runtime-relocate"),
            required_paths=("python.exe", "Scripts/conda-unpack.exe"),
            relocation=HealthCheck("Scripts/conda-unpack.exe", ()),
            health_check=HealthCheck("python.exe", ("-c", "pass")),
        )
        app = asset_for_bytes(
            app_zip,
            asset_id="app-relocation-test",
            component="app",
            variant="universal",
            required_path="spimaging/__init__.py",
            url="https://example.invalid/app-relocate",
        )

        class RecordingHealthRunner:
            def __init__(self) -> None:
                self.calls: list[tuple[Path, str | None]] = []

            def run(self, root: Path, check: HealthCheck | None) -> object:
                self.calls.append((root, check.executable if check else None))
                return object()

        runner = RecordingHealthRunner()
        manifest = ReleaseManifest(
            1,
            "SPImaging",
            "0.2.0-beta.1",
            "beta",
            "2026-08-23T00:00:00Z",
            True,
            (runtime, app),
            LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ()),
        )
        transport = MemoryTransport({runtime.parts[0].url: runtime_zip, app.parts[0].url: app_zip})
        with tempfile.TemporaryDirectory() as temporary:
            result = Provisioner(
                Path(temporary) / "install",
                transport,
                health_runner=runner,  # type: ignore[arg-type]
                desktop_smoke=noop_desktop_smoke,
            ).provision(manifest, "cpu", NvidiaCapability(False))
            runtime_calls = [call for call in runner.calls if call[1] is not None]
            self.assertEqual([call[1] for call in runtime_calls], ["Scripts/conda-unpack.exe", "python.exe"])
            self.assertTrue(all(call[0] == result.runtime_path for call in runtime_calls))

    def test_cuda_self_check_failure_falls_back_to_cpu(self) -> None:
        cpu_zip = make_zip({"python.exe": b"cpu"})
        cuda_zip = make_zip({"python.exe": b"cuda", "cuda-check.exe": b"check"})
        app_zip = make_zip({"spimaging/__init__.py": b"app"})
        cpu = asset_for_bytes(cpu_zip, url="https://example.invalid/cpu")
        cuda = replace(
            asset_for_bytes(
                cuda_zip,
                asset_id="runtime-cuda-test",
                variant="cuda",
                url="https://example.invalid/cuda",
            ),
            required_paths=("python.exe", "cuda-check.exe"),
            health_check=HealthCheck("cuda-check.exe", ()),
            min_nvidia_driver="550.0",
        )
        app = asset_for_bytes(
            app_zip,
            asset_id="app-fallback-test",
            component="app",
            variant="universal",
            required_path="spimaging/__init__.py",
            url="https://example.invalid/app-fallback",
        )

        class FailingCudaRunner:
            def run(self, root: Path, check: HealthCheck | None) -> object:
                if check and check.executable == "cuda-check.exe":
                    raise HealthCheckError("CUDA unavailable")
                return object()

        manifest = ReleaseManifest(
            1,
            "SPImaging",
            "0.2.0-beta.1",
            "beta",
            "2026-08-23T00:00:00Z",
            True,
            (cpu, cuda, app),
            LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ()),
        )
        content = {cpu.parts[0].url: cpu_zip, cuda.parts[0].url: cuda_zip, app.parts[0].url: app_zip}
        with tempfile.TemporaryDirectory() as temporary:
            result = Provisioner(
                Path(temporary) / "install",
                MemoryTransport(content),
                health_runner=FailingCudaRunner(),  # type: ignore[arg-type]
                desktop_smoke=noop_desktop_smoke,
            ).provision(manifest, "auto", NvidiaCapability(True, "551.0"))
            self.assertEqual(result.runtime_variant, "cpu")
            self.assertTrue(result.fallback)
            self.assertIn("CUDA 自检失败", result.reason)

    def test_app_update_reuses_independently_versioned_runtime(self) -> None:
        runtime_zip = make_zip({"python.exe": b"shared runtime"})
        app_v1_zip = make_zip({"spimaging/__init__.py": b"version = 1\n"})
        app_v2_zip = make_zip({"spimaging/__init__.py": b"version = 2\n"})
        runtime = replace(
            asset_for_bytes(runtime_zip, url="https://example.invalid/shared-runtime"),
            asset_id="runtime-cpu-0.2.0-runtime.1",
            version="0.2.0-runtime.1",
        )
        app_v1 = asset_for_bytes(
            app_v1_zip,
            asset_id="app-0.2.0-beta.1",
            component="app",
            variant="universal",
            required_path="spimaging/__init__.py",
            url="https://example.invalid/app-v1",
        )
        app_v2 = replace(
            asset_for_bytes(
                app_v2_zip,
                asset_id="app-0.2.0-beta.2",
                component="app",
                variant="universal",
                required_path="spimaging/__init__.py",
                url="https://example.invalid/app-v2",
            ),
            version="0.2.0-beta.2",
        )
        launch = LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ())
        first = ReleaseManifest(1, "SPImaging", "0.2.0-beta.1", "beta", "2026-08-23T00:00:00Z", True, (runtime, app_v1), launch)
        second = ReleaseManifest(1, "SPImaging", "0.2.0-beta.2", "beta", "2026-08-24T00:00:00Z", True, (runtime, app_v2), launch)
        transport = MemoryTransport(
            {
                runtime.parts[0].url: runtime_zip,
                app_v1.parts[0].url: app_v1_zip,
                app_v2.parts[0].url: app_v2_zip,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            provisioner = Provisioner(
                Path(temporary) / "install",
                transport,
                desktop_smoke=noop_desktop_smoke,
            )
            first_result = provisioner.provision(first, "cpu", NvidiaCapability(False))
            second_result = provisioner.provision(second, "cpu", NvidiaCapability(False))
            self.assertEqual(first_result.runtime_path, second_result.runtime_path)
            self.assertEqual(
                [url for url, _start in transport.starts].count(runtime.parts[0].url),
                1,
            )
            self.assertEqual((second_result.app_path / "spimaging" / "__init__.py").read_bytes(), b"version = 2\n")

    def test_combined_smoke_failure_restores_both_active_components(self) -> None:
        runtime_v1_zip = make_zip({"python.exe": b"runtime one"})
        app_v1_zip = make_zip({"spimaging/__init__.py": b"app one"})
        runtime_v2_zip = make_zip({"python.exe": b"runtime two"})
        app_v2_zip = make_zip({"spimaging/__init__.py": b"app two"})
        runtime_v1 = asset_for_bytes(runtime_v1_zip, asset_id="runtime-v1", url="https://example.invalid/runtime-v1")
        app_v1 = asset_for_bytes(app_v1_zip, asset_id="app-v1", component="app", variant="universal", required_path="spimaging/__init__.py", url="https://example.invalid/app-v1-transaction")
        runtime_v2 = replace(asset_for_bytes(runtime_v2_zip, asset_id="runtime-v2", url="https://example.invalid/runtime-v2"), version="0.2.0-beta.2")
        app_v2 = replace(asset_for_bytes(app_v2_zip, asset_id="app-v2", component="app", variant="universal", required_path="spimaging/__init__.py", url="https://example.invalid/app-v2-transaction"), version="0.2.0-beta.2")
        launch = LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ())
        first = ReleaseManifest(1, "SPImaging", "0.2.0-beta.1", "beta", "2026-08-23T00:00:00Z", True, (runtime_v1, app_v1), launch)
        second = ReleaseManifest(1, "SPImaging", "0.2.0-beta.2", "beta", "2026-08-24T00:00:00Z", True, (runtime_v2, app_v2), launch)
        content = {
            runtime_v1.parts[0].url: runtime_v1_zip,
            app_v1.parts[0].url: app_v1_zip,
            runtime_v2.parts[0].url: runtime_v2_zip,
            app_v2.parts[0].url: app_v2_zip,
        }

        def smoke(_runtime: Path, _app: Path, manifest: ReleaseManifest) -> object:
            if manifest.release_version == "0.2.0-beta.2":
                raise HealthCheckError("app import failed")
            return object()

        with tempfile.TemporaryDirectory() as temporary:
            provisioner = Provisioner(Path(temporary) / "install", MemoryTransport(content), desktop_smoke=smoke)
            first_result = provisioner.provision(first, "cpu", NvidiaCapability(False))
            before = provisioner.manager.snapshot_state().state
            with self.assertRaisesRegex(HealthCheckError, "app import failed"):
                provisioner.provision(second, "cpu", NvidiaCapability(False))
            self.assertEqual(provisioner.manager.snapshot_state().state, before)
            self.assertEqual(provisioner.manager.active_path("runtime"), first_result.runtime_path)
            self.assertEqual(provisioner.manager.active_path("app"), first_result.app_path)

    def test_disk_preflight_checks_cache_and_install_volumes_before_download(self) -> None:
        runtime_zip = make_zip({"python.exe": b"runtime"})
        app_zip = make_zip({"spimaging/__init__.py": b"app"})
        runtime = asset_for_bytes(runtime_zip, url="https://example.invalid/preflight-runtime")
        app = asset_for_bytes(app_zip, asset_id="preflight-app", component="app", variant="universal", required_path="spimaging/__init__.py", url="https://example.invalid/preflight-app")
        manifest = ReleaseManifest(1, "SPImaging", "0.2.0-beta.1", "beta", "2026-08-23T00:00:00Z", True, (runtime, app), LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ()))
        transport = MemoryTransport({runtime.parts[0].url: runtime_zip, app.parts[0].url: app_zip})
        usage = type("Usage", (), {})
        with tempfile.TemporaryDirectory() as temporary:
            provisioner = Provisioner(Path(temporary) / "install", transport, desktop_smoke=noop_desktop_smoke)
            with patch("launcher.bootstrap.shutil.disk_usage", return_value=usage()) as mocked:
                mocked.return_value.free = 1
                with self.assertRaisesRegex(DownloadError, "download cache"):
                    provisioner.provision(manifest, "cpu", NvidiaCapability(False))
            self.assertEqual(transport.starts, [])
            with patch("launcher.bootstrap.shutil.disk_usage") as mocked:
                cache_usage = usage()
                cache_usage.free = 10**12
                install_usage = usage()
                install_usage.free = 1
                mocked.side_effect = [cache_usage, install_usage]
                with self.assertRaisesRegex(DownloadError, "installation/staging and rollback"):
                    provisioner.provision(manifest, "cpu", NvidiaCapability(False))
            self.assertEqual(transport.starts, [])


class UpdateStateTests(unittest.TestCase):
    def test_desktop_settings_control_runtime_cache_and_update_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            install_root = Path(temporary)
            cache_dir = install_root / "custom cache"
            (install_root / "settings.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "device": "cpu",
                        "cache_dir": str(cache_dir),
                        "update_checks": False,
                        "ignored": "value",
                    }
                ),
                encoding="utf-8",
            )
            args = build_launcher_parser().parse_args(
                ["--install-root", str(install_root), "--headless"]
            )
            preferences = load_desktop_preferences(install_root)
            self.assertEqual(preferences["cache_dir"], str(cache_dir))
            self.assertEqual(runtime_preference(args), "cpu")
            self.assertTrue(_updates_disabled(args))
            self.assertEqual(args.channel, "beta")
            self.assertIsNone(args.manifest_url)
            self.assertIn("/windows-beta/", DEFAULT_BETA_MANIFEST_URL)

            explicit = build_launcher_parser().parse_args(
                ["--install-root", str(install_root), "--runtime", "cuda", "--headless"]
            )
            self.assertEqual(runtime_preference(explicit), "cuda")
            explicit.check_now = True
            self.assertFalse(_updates_disabled(explicit))

    def test_manifest_update_rejects_plain_http_before_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ManifestCache(Path(temporary))
            transport = MemoryTransport({})
            with self.assertRaisesRegex(LauncherError, "HTTPS"):
                cache.resolve("http://example.invalid/manifest.json", transport)
            self.assertEqual(transport.starts, [])

    def test_update_checks_are_limited_to_once_per_24_hours(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ManifestCache(Path(temporary))
            now = datetime(2026, 8, 23, tzinfo=timezone.utc)
            self.assertTrue(cache.should_check(now=now))
            cache.record_check(now=now)
            self.assertFalse(cache.should_check(now=now + timedelta(hours=23, minutes=59)))
            self.assertTrue(cache.should_check(now=now + timedelta(hours=24)))
            self.assertTrue(cache.should_check(now=now, force=True))

    def test_failed_online_check_with_cache_is_throttled_and_downgrades_are_ignored(self) -> None:
        runtime_zip = make_zip({"python.exe": b"runtime"})
        app_zip = make_zip({"spimaging/__init__.py": b"app"})
        runtime = asset_for_bytes(runtime_zip)
        app = asset_for_bytes(app_zip, asset_id="cached-app", component="app", variant="universal", required_path="spimaging/__init__.py")
        launch = LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ())
        current = ReleaseManifest(1, "SPImaging", "0.2.0-beta.2", "beta", "2026-08-24T00:00:00Z", True, (replace(runtime, version="0.2.0-runtime.1"), replace(app, version="0.2.0-beta.2")), launch)
        older = ReleaseManifest(1, "SPImaging", "0.2.0-beta.1", "beta", "2026-08-23T00:00:00Z", True, (replace(runtime, version="0.2.0-runtime.1"), app), launch)
        current_raw = json.dumps(manifest_to_dict(current)).encode("utf-8")
        older_raw = json.dumps(manifest_to_dict(older)).encode("utf-8")
        url = "https://example.invalid/release-manifest.json"
        with tempfile.TemporaryDirectory() as temporary:
            cache = ManifestCache(Path(temporary))
            cache.store(current_raw, current)
            resolved, fetched, message = cache.resolve(
                url,
                MemoryTransport({url: older_raw}),
                force_check=True,
                expected_channel="beta",
            )
            self.assertFalse(fetched)
            self.assertEqual(resolved, current)
            self.assertIn("downgrade", message)
            self.assertIsNotNone(cache.last_checked())
            self.assertFalse(_is_update(ProvisionResult("0.2.0-beta.2", Path(), Path(), "cpu", False, ""), older))

            # A stale candidate cache must not bypass the newer active release
            # merely because the 24-hour network interval has not elapsed.
            cache.mark_active(current_raw, current)
            cache.store(older_raw, older)
            cache.record_check()
            offline = MemoryTransport({})
            with self.assertRaisesRegex(LauncherError, "downgrade"):
                cache.resolve(url, offline, expected_channel="beta")
            self.assertEqual(offline.starts, [])

    def test_signed_manifest_is_verified_against_launcher_owned_thumbprint_before_cache(self) -> None:
        archive = make_zip({"python.exe": b"runtime"})
        thumbprint = "A" * 40
        requirement = SignatureRequirement(
            "cms-detached",
            True,
            signer_thumbprint=thumbprint,
            file_name="asset.zip.p7s",
            url="https://example.invalid/asset.zip.p7s",
            size=3,
            sha256="b" * 64,
        )
        runtime = asset_for_bytes(archive, signature=requirement)
        app = asset_for_bytes(
            make_zip({"spimaging/__init__.py": b"app"}),
            asset_id="signed-app",
            component="app",
            variant="universal",
            required_path="spimaging/__init__.py",
            signature=requirement,
        )
        manifest = ReleaseManifest(1, "SPImaging", "0.2.0-beta.1", "beta", "2026-08-23T00:00:00Z", False, (runtime, app), LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ()))
        raw = json.dumps(manifest_to_dict(manifest)).encode("utf-8")
        signature = b"trusted manifest signature"
        url = "https://example.invalid/release-manifest.json"
        verified: list[tuple[bytes, bytes, str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            cache: ManifestCache

            def verify(content: bytes, cms: bytes, expected: str) -> None:
                if not verified:
                    self.assertFalse(cache.manifest_path.exists())
                verified.append((content, cms, expected))

            policy = ManifestTrustPolicy.signed(thumbprint, detached_verifier=verify)
            cache = ManifestCache(Path(temporary), trust_policy=policy)
            resolved, fetched, _message = cache.resolve(
                url,
                MemoryTransport({url: raw, url + ".p7s": signature}),
            )
            self.assertTrue(fetched)
            self.assertEqual(resolved, manifest)
            self.assertEqual(verified, [(raw, signature, thumbprint)])
            self.assertEqual(cache.load(), manifest)
            self.assertEqual(verified[-1], (raw, signature, thumbprint))

            attacker_requirement = replace(requirement, signer_thumbprint="C" * 40)
            attacker_manifest = replace(
                manifest,
                assets=(replace(runtime, signature=attacker_requirement), app),
            )
            attacker_raw = json.dumps(manifest_to_dict(attacker_manifest)).encode("utf-8")
            verified_before = len(verified)
            with self.assertRaisesRegex(SignatureError, "not pinned"):
                policy.verify_manifest(attacker_raw, attacker_manifest, b"attacker cms")
            self.assertEqual(len(verified), verified_before)

    def test_unsigned_beta_and_formal_signed_trust_modes_are_isolated(self) -> None:
        # Use a valid unsigned manifest from the provisioning helper rather
        # than relying on a structurally invalid control document.
        runtime_zip = make_zip({"python.exe": b"runtime"})
        app_zip = make_zip({"spimaging/__init__.py": b"app"})
        runtime = asset_for_bytes(runtime_zip)
        app = asset_for_bytes(app_zip, asset_id="unsigned-app", component="app", variant="universal", required_path="spimaging/__init__.py")
        manifest = ReleaseManifest(1, "SPImaging", "0.2.0-beta.1", "beta", "2026-08-23T00:00:00Z", True, (runtime, app), LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ()))
        raw = json.dumps(manifest_to_dict(manifest)).encode("utf-8")
        ManifestTrustPolicy.unsigned_beta().verify_manifest(raw, manifest, None)
        with self.assertRaisesRegex(SignatureError, "signed launcher refuses"):
            ManifestTrustPolicy.signed("A" * 40, detached_verifier=lambda *_: None).verify_manifest(raw, manifest, b"sig")

    def test_interprocess_lock_fails_clearly_for_second_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata" / "launcher.lock"
            with InterProcessLock(path, purpose="测试单实例"):
                with self.assertRaisesRegex(LauncherError, "另一 SPImaging 实例"):
                    InterProcessLock(path, purpose="测试单实例").acquire()


class PrivateRuntimeEnvironmentTests(unittest.TestCase):
    def test_desktop_command_removes_user_python_and_conda_shadowing(self) -> None:
        from launcher.bootstrap import build_desktop_command

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            app = root / "app"
            (runtime / "Scripts").mkdir(parents=True)
            (runtime / "Library" / "bin").mkdir(parents=True)
            app.mkdir()
            (runtime / "python.exe").write_bytes(b"python")
            manifest = ReleaseManifest(
                1,
                "SPImaging",
                "0.2.0-beta.1",
                "beta",
                "2026-08-23T00:00:00Z",
                True,
                (),
                LaunchSpec("pythonw.exe", "python.exe", "spimaging.desktop", ()),
            )
            result = ProvisionResult("0.2.0-beta.1", runtime, app, "cpu", False, "")
            with patch.dict(
                os.environ,
                {
                    "PYTHONHOME": "C:/host-python",
                    "PYTHONPATH": "C:/untrusted-modules",
                    "CONDA_PREFIX": "C:/host-conda",
                    "CONDA_DEFAULT_ENV": "host",
                    "VIRTUAL_ENV": "C:/venv",
                },
                clear=False,
            ):
                _command, environment = build_desktop_command(manifest, result)
            self.assertEqual(environment["PYTHONPATH"], str(app))
            self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
            self.assertNotIn("PYTHONHOME", environment)
            self.assertNotIn("CONDA_PREFIX", environment)
            self.assertNotIn("CONDA_DEFAULT_ENV", environment)
            self.assertNotIn("VIRTUAL_ENV", environment)
            self.assertTrue(environment["PATH"].startswith(str(runtime)))


if __name__ == "__main__":
    unittest.main()
