# SPImaging Windows release inputs

This directory contains reproducible **build inputs**, not committed release
binaries.  Nothing here uploads a GitHub release unless a maintainer explicitly
dispatches the release workflow with `publish=true` from a `v*` tag.

## Prerequisites on a Windows build host

- Miniconda/Miniforge and `conda-lock`
- `conda-pack`
- Python 3.10+ and PyInstaller (launcher build only)
- Inno Setup 6 (`ISCC.exe`) for `SPImaging-Setup.exe`
- Optional: an existing Authenticode certificate and Windows SDK `signtool.exe`

No certificate is created by these scripts.  When signing inputs are absent,
the output name and manifest remain explicitly marked `unsigned-beta`.

## Reproducible sequence

```powershell
packaging/scripts/New-RuntimeLocks.ps1
packaging/scripts/Build-Runtime.ps1 -Variant cpu
packaging/scripts/Build-Runtime.ps1 -Variant cuda
packaging/scripts/Build-AppAsset.ps1
packaging/scripts/Build-Launcher.ps1
packaging/scripts/New-ReleaseManifest.ps1 -Version 0.2.0-beta.1 -BaseUrl https://github.com/ewellchen/SPImaging/releases/download/v0.2.0-beta.1
packaging/scripts/Build-Installer.ps1 -Version 0.2.0-beta.1
```

`New-RuntimeLocks.ps1` is the only dependency-solving step and runs on the
build host.  End-user machines receive the resulting `conda-pack` ZIP and never
run Conda or a solver.  Release builds archive both generated lock files and
their SHA-256 digests next to the assets.

Use the following offline dry-run before publishing:

```powershell
python packaging/scripts/verify_release_manifest.py packaging/out/spimaging-release-manifest.json --asset-dir packaging/out
```

The application asset builder includes only tracked `spimaging/`, synthetic
`public_demo/`, `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, and `SBOM.md`
paths. It rejects the private `example_data/` and legacy `demo_checkpoint/`
trees.

`Build-Installer.ps1` has two deterministic signing paths. Passing
`-InnoSignToolCommand` lets Inno emit `SPImaging-Setup.exe` directly. Passing
`-SigningPfxPath` and `-CertificateThumbprint` compiles the explicitly named
`SPImaging-Setup-unsigned-beta.exe`, Authenticode-signs that exact file through
`Sign-Artifact.ps1`, then atomically renames it to `SPImaging-Setup.exe`. With no
signing input, only the `-unsigned-beta` filename is produced.
