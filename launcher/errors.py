"""Launcher-specific errors with messages suitable for a small bootstrap UI."""


class LauncherError(RuntimeError):
    """Base class for a controlled launcher failure."""


class ManifestError(LauncherError):
    """The remote release description is missing or invalid."""


class DownloadError(LauncherError):
    """An asset could not be downloaded or did not match its metadata."""


class SignatureError(LauncherError):
    """A required publisher signature is absent or invalid."""


class ExtractionError(LauncherError):
    """An archive failed closed during validation or extraction."""


class HealthCheckError(LauncherError):
    """An unpacked runtime/app failed its declared health check."""


class ActivationError(LauncherError):
    """A staged release could not be activated or rolled back."""
