# SPImaging source SBOM

| Field | Value |
| --- | --- |
| Document version | 2 |
| Document date | 2026-08-24 |
| Product | SPImaging 0.2.0-beta.1 |
| Target | Windows 10/11 x64 desktop public beta |
| Project license | Apache-2.0 |
| Public demo asset license | CC0-1.0 |
| Relationship | Components below are direct dependencies or build tools of SPImaging |

This human-readable software bill of materials describes the source release and
the runtime evidence that exists for this beta. The Conda lock files remain the
authoritative component-level list of package URLs and package hashes.

| Component | Package URL / identity | Version | Scope | License |
| --- | --- | --- | --- | --- |
| SPImaging | `pkg:pypi/spimaging@0.2.0-beta.1` | 0.2.0-beta.1 | application | Apache-2.0 |
| Python | `pkg:generic/python` | 3.10.21 (CPU archive) | runtime | Python-2.0 |
| NumPy | `pkg:pypi/numpy` | 1.26.4 (CPU archive) | runtime | BSD-3-Clause |
| SciPy | `pkg:pypi/scipy` | 1.15.2 (CPU archive) | runtime | BSD-3-Clause |
| h5py | `pkg:pypi/h5py` | 3.16.0 (CPU archive) | runtime | BSD-3-Clause |
| HDF5 | `pkg:generic/hdf5` | 1.14.6 (CPU archive) | runtime/native | BSD-3-Clause |
| imageio | `pkg:pypi/imageio` | 2.37.0 (CPU archive) | runtime | BSD-2-Clause |
| scikit-image | `pkg:pypi/scikit-image` | 0.25.2 (CPU archive) | runtime | BSD-3-Clause |
| Matplotlib | `pkg:pypi/matplotlib` | 3.10.9 (CPU archive) | runtime | LicenseRef-Matplotlib |
| tqdm | `pkg:pypi/tqdm` | 4.69.0 (CPU archive) | runtime | MPL-2.0 AND MIT |
| OpenCV | `pkg:conda/conda-forge/opencv` | 4.11.0 (CPU archive) | runtime | Apache-2.0 |
| PyTorch | `pkg:pypi/torch` | 2.5.1 CPU build (CPU archive and current CUDA lock) | runtime | BSD-3-Clause |
| DeepInverse | `pkg:pypi/deepinv` | 0.4.1 (CPU archive) | runtime | BSD-3-Clause |
| PySide6 / Qt | `pkg:conda/conda-forge/pyside6` | 6.9.3 (CPU archive) | desktop runtime | LGPL-3.0-only |
| conda-pack | `pkg:pypi/conda-pack` | 0.9.2 | build | BSD-3-Clause |
| setuptools | `pkg:pypi/setuptools` | `>=68` | build | MIT |
| pytest | `pkg:pypi/pytest` | resolved at build | test | MIT |
| Inno Setup | `pkg:generic/inno-setup` | resolved at build | installer build | LicenseRef-Inno-Setup |

## Runtime evidence snapshot

| Evidence | Status |
| --- | --- |
| CPU lock | `environment-cpu.conda-lock.yml`; SHA-256 `799cd097068459682ca897eb6febd9d3284111a8741679ff0bae97fe53cd6640`; 202 Conda + 7 pip records |
| CPU archive | `spimaging-runtime-cpu-0.2.0-runtime.1.zip`; 867,690,922 bytes; SHA-256 `8914897c45c710bf155176593912318b7a21efa0c4d02da2fa7d201b5c10d766`; build health verified PyTorch 2.5.1, PySide6 6.9.3 and DeepInverse 0.4.1 before packing |
| CUDA lock | `environment-cuda.conda-lock.yml`; SHA-256 `541dfe2c4198ca4c57cda158e4a2a87359ecc4f9f011470af6af719449dbf5d5`; 235 Conda + 7 pip records; locked only, with no built or validated CUDA archive |

**Release blocker:** the CUDA lock resolves a CPU-only PyTorch build and the runtime GPL/LGPL redistribution obligations are not yet fully closed, so this unsigned CPU build is for controlled beta testing rather than public release.

## Generated release assets

The public demo dataset and Simple3D checkpoint are generated components, not
third-party packages. Their individual file hashes, input relationships,
generator hash, parameters, framework version, and provenance are recorded in
`public_demo/manifest.json`. CPU/CUDA private runtime and installer manifests
are owned by the release build and must be attached to the release SBOM.
