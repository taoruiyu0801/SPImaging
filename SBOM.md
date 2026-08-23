# SPImaging source SBOM

| Field | Value |
| --- | --- |
| Document version | 1 |
| Document date | 2026-08-23 |
| Product | SPImaging 0.2.0-beta.1 |
| Target | Windows 10/11 x64 desktop public beta |
| Project license | Apache-2.0 |
| Public demo asset license | CC0-1.0 |
| Relationship | Components below are direct dependencies or build tools of SPImaging |

This human-readable software bill of materials describes the source release.
Versions shown as ranges are resolved and locked separately for CPU and CUDA
runtime archives. A release SBOM must replace `resolved at build` with exact
versions, package URLs, archive SHA-256 values, and transitive/native components
before publication.

| Component | Package URL / identity | Version | Scope | License |
| --- | --- | --- | --- | --- |
| SPImaging | `pkg:pypi/spimaging@0.2.0-beta.1` | 0.2.0-beta.1 | application | Apache-2.0 |
| Python | `pkg:generic/python` | 3.10.x locked at build | runtime | PSF-2.0 |
| NumPy | `pkg:pypi/numpy` | `<2` | runtime | BSD-3-Clause |
| SciPy | `pkg:pypi/scipy` | resolved at build | runtime | BSD-3-Clause |
| h5py | `pkg:pypi/h5py` | resolved at build | runtime | BSD-3-Clause |
| HDF5 | `pkg:generic/hdf5` | resolved at build | runtime/native | HDF5 |
| imageio | `pkg:pypi/imageio` | resolved at build | runtime | BSD-2-Clause |
| scikit-image | `pkg:pypi/scikit-image` | resolved at build | runtime | BSD-3-Clause |
| Matplotlib | `pkg:pypi/matplotlib` | resolved at build | runtime | LicenseRef-Matplotlib |
| tqdm | `pkg:pypi/tqdm` | resolved at build | runtime | MPL-2.0 AND MIT |
| opencv-python | `pkg:pypi/opencv-python` | resolved at build | runtime | Apache-2.0 |
| PyTorch | `pkg:pypi/torch` | CPU/CUDA variant locked at build | runtime | BSD-3-Clause |
| DeepInverse | `pkg:pypi/deepinv` | resolved at build | optional runtime | BSD-3-Clause |
| PySide6 | `pkg:pypi/pyside6` | resolved at build | desktop runtime | LGPL-3.0-only OR GPL-3.0-only OR LicenseRef-Qt-Commercial |
| conda-pack | `pkg:pypi/conda-pack` | resolved at build | build | BSD-3-Clause |
| setuptools | `pkg:pypi/setuptools` | `>=68` | build | MIT |
| pytest | `pkg:pypi/pytest` | resolved at build | test | MIT |
| Inno Setup | `pkg:generic/inno-setup` | resolved at build | installer build | LicenseRef-Inno-Setup |

## Generated release assets

The public demo dataset and Simple3D checkpoint are generated components, not
third-party packages. Their individual file hashes, input relationships,
generator hash, parameters, framework version, and provenance are recorded in
`public_demo/manifest.json`. CPU/CUDA private runtime and installer manifests
are owned by the release build and must be attached to the release SBOM.
