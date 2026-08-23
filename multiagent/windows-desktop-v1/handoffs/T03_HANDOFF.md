# T03 Handoff: Launcher And Release Packaging

## Status
- State: Complete (source implementation and reproducible build inputs)

## What Changed
- Added a standard-library bootstrap library with strict release-manifest v1 validation, CPU/CUDA selection, NVIDIA driver probing, resumable HTTPS downloads, per-part and assembled SHA-256 checks, optional pinned Authenticode/detached-CMS verification, ZIP bomb/path/symlink/device-name defenses, disk preflight, final-prefix conda relocation, health checks, atomic activation state, quarantine repair, and one-step rollback.
- Added a minimal Chinese Tk bootstrap entry (`python -m launcher`) with first-install progress, update confirmation, 24-hour update checks, verified offline fallback, Auto/CUDA/CPU selection, repair/headless/dry-run modes, and private-runtime desktop launch without a console dependency.
- Added deterministic application packaging (including mandatory license/third-party/SBOM documents), split-asset and release-manifest generation/verification tools, conda-lock/conda-pack CPU and CUDA build scripts, conditional signing hooks, a PyInstaller spec, and a per-user Inno Setup definition that does not modify PATH or remove user runtimes/results on uninstall. Both runtime inputs include matched PyTorch/torchvision, PySide6, and DeepInverse; activation health imports the GUI and single-surface dependencies.
- Added Windows packaging CI and an explicitly gated release-candidate workflow. GitHub release upload requires all of: manual dispatch, `publish=true`, `build_runtime=true`, exact `v<version>` tag, protected release environment approval, and all signing secrets. Default dispatch only builds local Actions artifacts.
- Added offline unit coverage for manifest validation, selection/fallback, resume behavior, bad hashes/truncation, required signature failure, malicious archives, activation/rollback/repair, final-path conda relocation, CUDA self-check fallback, provisioning, and update throttling.

## Files Touched
- `launcher/**`
- `packaging/**`
- `.github/workflows/windows-packaging-check.yml`
- `.github/workflows/windows-release-candidate.yml`
- `tests/test_launcher.py`
- `tests/test_release_manifest.py`
- This handoff and `smoke-tests/T03_SMOKE_TEST.md`

## Smoke Test Result
- PASS: `conda run -n spimaging python -m pytest -q tests/test_launcher.py tests/test_release_manifest.py`
- Result: `28 passed, 12 subtests passed in 0.37s` (latest repeated run remained green).
- PASS: example release-manifest structural dry-run, launcher source dry-run, Python compileall, and static parse of all seven PowerShell build scripts.
- Integration observation: a broad `pytest -q` run reached `128 passed, 101 subtests passed` with two failures outside T03 ownership: the generated Day 13-14 CLI parameter table is stale after new recovery flags, and `tests/test_smoke.py` still expects package version `0.1.0` while integration now exposes `0.2.0-beta.1`.

## Decisions Made
- Manifest-controlled network URLs are HTTPS-only and reject embedded credentials/downgrade redirects. CPU runtime and universal app assets are mandatory; CUDA remains optional and always has an explained CPU fallback.
- Unsigned artifacts are legal only when the manifest explicitly marks a beta. A stable manifest requires a pinned publisher signature on every asset. Runtime/app ZIPs use detached CMS signatures because ZIP files cannot carry Authenticode; launcher/installer PE files use Authenticode.
- Split parts are individually hashed, then the reconstructed archive is independently size/hash checked before signature verification and extraction.
- `conda-unpack` runs only after the staging directory has been atomically placed at its final release path, but before activation state changes; this avoids embedding a temporary prefix. Any relocation/health failure restores the prior state.
- The last active manifest is stored separately from the newest checked manifest, so declining an update or losing connectivity cannot accidentally launch new metadata against old files.
- Runtime dependency solving is build-host-only. `New-RuntimeLocks.ps1` produces and hashes win-64 conda locks; `Build-Runtime.ps1` refuses to build without those locks, then emits conda-pack ZIPs and <=1.8 GiB signed-part inputs.
- Signed installer finalization is deterministic: `Build-Installer.ps1` either delegates an explicit sign tool to Inno, or compiles the exact `-unsigned-beta` path, Authenticode-signs it with the pinned supplied PFX, and renames it to `SPImaging-Setup.exe`. Without signing inputs it never emits the production filename.

## Open Issues For Downstream Context
- This machine does not have PyInstaller, conda-lock, conda-pack, or Inno Setup (`ISCC.exe`), so no real launcher EXE, multi-GiB runtime, or installer was generated here. The scripts fail with the exact missing prerequisite; T_FINAL should run them on the Windows release builder.
- No Authenticode certificate/secrets are available. Current sample output is deliberately `unsigned_beta`; a public signed upload remains blocked by the workflow until PFX/password/thumbprint secrets and release-environment approval exist.
- The launcher contract targets `spimaging.desktop`; that module was not present when T03 finished and must be supplied/verified by T02/T_FINAL.
- Clean Windows VM, actual NVIDIA/CUDA, SmartScreen, proxy, and uninstall-preservation acceptance require external-machine validation by T_FINAL; unit tests use local temporary assets only.
- Generated conda lock files are build artifacts rather than checked-in files in this task. Archive the generated locks and `.sha256` records with each release candidate for reproducibility/audit.
- T_FINAL must regenerate the CLI parameter reference and update the old package-version smoke assertion; these were the only failures in T03's broad integration test and are outside this write boundary.

## Boundary Notes
- Stayed within write boundary: Yes. No root manifest, application algorithm, GUI/appcore, license, user dirty file, or other task-owned file was edited. No files were staged or committed and no remote release action was invoked.

## Suggested Next Checks
- Re-run `conda run -n spimaging python -m pytest -q tests/test_launcher.py tests/test_release_manifest.py` after integration.
- Run `python packaging/scripts/verify_release_manifest.py <manifest> --asset-dir <release-assets>` against the real CPU/CUDA/app archives.
- On a clean Windows build host, run the sequence in `packaging/README.md`, verify `SPImaging-Setup.exe`/`SPImaging-Setup-unsigned-beta.exe`, then test first online bootstrap and second offline launch as a standard user.
- Verify detached CMS and Authenticode thumbprint pinning with the real release certificate before enabling the protected publish job.
