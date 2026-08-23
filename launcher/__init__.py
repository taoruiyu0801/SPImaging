"""Safe bootstrap launcher for the SPImaging Windows desktop application.

The launcher intentionally depends only on the Python standard library so it can
be frozen into a small, self-contained executable.  The scientific runtime is
downloaded separately and is never dependency-solved on an end user's machine.
"""

from .errors import (
    ActivationError,
    DownloadError,
    ExtractionError,
    HealthCheckError,
    LauncherError,
    ManifestError,
    SignatureError,
)
from .manifest import ReleaseManifest

__all__ = [
    "ActivationError",
    "DownloadError",
    "ExtractionError",
    "HealthCheckError",
    "LauncherError",
    "ManifestError",
    "ReleaseManifest",
    "SignatureError",
]

__version__ = "0.2.0-beta.1"
