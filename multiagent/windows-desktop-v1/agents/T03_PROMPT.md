# T03 Agent Prompt: Launcher And Release Packaging

Read the constitution, task breakdown, and global acceptance first. No upstream handoff is required.

## Objective

Implement the safe bootstrap/update library, minimal launcher UI/entry, runtime/app release manifests, conda-pack build scripts, Inno Setup script, and non-publishing-by-default GitHub workflows.

## Allowed Write Boundary

- `launcher/**`
- `packaging/**`
- `.github/workflows/**`
- `tests/test_launcher.py`, `tests/test_release_manifest.py`
- T03 handoff and smoke files

## Forbidden Changes

- Do not edit application algorithms, GUI, appcore, root dependency manifests, licenses, or user dirty files.
- Do not upload releases, create credentials, or invoke destructive uninstall logic.

## Required Deliverables

- Release manifest parser/validation, CPU/CUDA selection, resuming downloader, safe extraction, SHA-256 validation, staging/activation/rollback and health-check hooks.
- Per-user Inno Setup definition and launcher/PyInstaller build inputs.
- Locked-runtime/app build and split-asset scripts plus signing hooks.
- Unit tests using local temporary files/mock transport only.

## Smoke Test

```powershell
conda run -n spimaging python -m pytest -q tests/test_launcher.py tests/test_release_manifest.py
```
