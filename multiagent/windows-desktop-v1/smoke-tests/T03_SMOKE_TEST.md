# T03 Smoke Test: Launcher And Release Packaging

## Test Surface
Manifest selection, resume download, hash/safe extraction, activation and rollback.

## Procedure
1. Run launcher/release pytest files with temporary local assets.
2. Run packaging manifest dry-run.

## Expected Result
- Tests pass without network, registry writes or release upload.

## Actual Result
- PASS (2026-08-23, Asia/Shanghai).

```text
conda run -n spimaging python -m pytest -q tests/test_launcher.py tests/test_release_manifest.py
............................                                 [100%]
28 passed, 12 subtests passed in 0.37s
```

- `conda run -n spimaging python packaging/scripts/verify_release_manifest.py packaging/manifests/release-manifest.unsigned-beta.example.json`
  - `OK schema=1 version=0.2.0-beta.1 assets=2`
- `conda run -n spimaging python -m launcher --manifest-file packaging/manifests/release-manifest.unsigned-beta.example.json --dry-run --runtime auto`
  - Parsed version `0.2.0-beta.1`, selected CPU, and emitted an explicit missing-CUDA fallback reason.
- `conda run -n spimaging python -m compileall -q launcher packaging/scripts tests/test_launcher.py tests/test_release_manifest.py`
  - Exit code 0, no output.
- PowerShell `[scriptblock]::Create(...)` parsed all seven `packaging/scripts/*.ps1` files successfully.
- Tests used only temporary local ZIPs/in-memory transports; no network, registry, installer, credentials, release upload, or user data mutation occurred.
- External build note: PyInstaller, conda-lock, conda-pack, and Inno Setup are not installed on this machine, so binary/runtime/installer generation was intentionally deferred to T_FINAL's release host.
- Broad integration check: `conda run -n spimaging python -m pytest -q` produced `2 failed, 128 passed, 101 subtests passed`. Both failures are outside T03 files: stale generated CLI parameter tables after parallel recovery-flag work, and the old `0.1.0` assertion in `tests/test_smoke.py` after the integrated version became `0.2.0-beta.1`. T03's required smoke command remains fully green.
