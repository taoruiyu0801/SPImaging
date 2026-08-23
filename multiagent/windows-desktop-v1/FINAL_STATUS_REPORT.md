# Windows Beta Final Status

## Runnable delivery

- Version: `0.2.0-beta.1`, unsigned CPU beta.
- Classmate bundle: `C:\Users\32499\Desktop\record_of_SPI\Windows_Beta_20260824_CPU`.
- One-click offline path: installer -> local SHA-256 verified runtime/app provisioning -> PySide6 desktop.
- Locked CPU runtime: Python 3.10.21, PyTorch 2.5.1 CPU, PySide6 6.9.3, DeepInverse 0.4.1.

## Final verification

- Full source regression: `203 passed, 1 skipped, 130 subtests passed`.
- Launcher/release target: `46 passed, 26 subtests passed`.
- Compileall and `git diff --check`: pass.
- Real isolated runtime: desktop smoke, Simple3D prediction, and four-sample evaluation pass.
- Real recovery: training cancel/resume and same-run generation resume pass with structured batch/sample events.
- Frozen launcher offline manifest/asset-directory dry-run: pass.

## Release artifacts

- Runtime ZIP: 867,690,922 bytes; SHA-256 `8914897c45c710bf155176593912318b7a21efa0c4d02da2fa7d201b5c10d766`.
- App ZIP: SHA-256 `3832d8466d1db6fd1610fef1b2ea501e3ff323a8ea5a935ba2430d555470b83f`.
- Launcher: SHA-256 `2dcb821aa4d3d95f348cc7bc4dc3a593532a37ceed3146c0c5966c96ad2dd323`.
- Installer: SHA-256 `8f053ef22a972df2dd255dac470534f330aff2773a5e3fa98ddb116b3ba77b42`.

## Deferred

- CUDA runtime is withheld because the current CUDA lock resolves CPU-only PyTorch.
- Authenticode/CMS signing and SmartScreen reputation require a purchased certificate.
- Clean second-machine, NVIDIA, proxy, and SmartScreen acceptance remains external.
- Same-version repair hard-kill recovery and immutable rerun of an existing release tag remain follow-ups.
