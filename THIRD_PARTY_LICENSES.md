# Third-party software notices

SPImaging application source is Apache-2.0; this file inventories direct
third-party components expected by the source and Windows public runtime. Each
component remains under its own license. Release packaging must retain the
license files installed by each locked distribution, including notices for
their bundled native libraries.

| Component | Purpose / distribution scope | License | Upstream |
| --- | --- | --- | --- |
| Python | Private application runtime | PSF-2.0 | https://www.python.org/ |
| NumPy | Array and NPZ processing | BSD-3-Clause (binary builds may bundle separately licensed BLAS/runtime components) | https://numpy.org/ |
| SciPy | Scientific routines | BSD-3-Clause | https://scipy.org/ |
| h5py / HDF5 | HDF5 dataset input | BSD-3-Clause / HDF5 license | https://www.h5py.org/ |
| imageio | Image input | BSD-2-Clause | https://imageio.readthedocs.io/ |
| scikit-image | Image processing | BSD-3-Clause | https://scikit-image.org/ |
| Matplotlib | Plots and result export | Matplotlib License (PSF-derived) | https://matplotlib.org/ |
| tqdm | Progress display | MPL-2.0 AND MIT | https://tqdm.github.io/ |
| OpenCV / opencv-python | Image processing | Apache-2.0; wheel may include separately licensed third-party components | https://opencv.org/ |
| PyTorch | Training and inference | BSD-3-Clause; binaries include separately licensed native components | https://pytorch.org/ |
| DeepInverse | Optional single-surface simulation | BSD-3-Clause | https://deepinv.github.io/ |
| PySide6 / Qt for Python | Desktop GUI and Qt runtime | LGPL-3.0-only OR GPL-3.0-only OR commercial Qt terms, as applicable | https://doc.qt.io/qtforpython-6/ |
| conda-pack | Reproducible private-runtime archives (build only) | BSD-3-Clause | https://conda.github.io/conda-pack/ |
| setuptools | Python application build | MIT | https://setuptools.pypa.io/ |
| pytest | Test tooling; not required at application runtime | MIT | https://pytest.org/ |
| Inno Setup | Windows installer compiler and installer support code | Inno Setup License | https://jrsoftware.org/isinfo.php |

## Qt redistribution note

Public beta builds using the LGPL option must keep Qt/PySide6 libraries as
separate, replaceable dynamic libraries, include the corresponding LGPL text
and Qt notices from the selected binary distribution, disclose modifications
(none are intended), and provide the applicable source-code offer/instructions
required by that distribution. A commercial Qt license may impose different
terms. The release engineer must verify the selected runtime before publishing.

## Native and transitive components

NumPy, SciPy, OpenCV, PyTorch, Qt, HDF5, and their binary packages can bundle
native libraries whose exact set depends on the locked CPU or CUDA runtime.
`SBOM.md` is the source-level inventory, not a substitute for the final archive
scan. The release process must export the resolved package list and retain every
license/notice from the actual archives. CUDA/NVIDIA redistribution is governed
by the NVIDIA CUDA Toolkit EULA and the redistributable-file list for the exact
runtime version; GPU drivers are never bundled.

License identifiers and links are informational and are not legal advice. The
license texts contained in the resolved distributions control in case of a
conflict.
