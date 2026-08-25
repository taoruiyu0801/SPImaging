"""SPImaging PySide6 workbench.

The package itself remains importable when the optional GUI dependency is not
installed.  Calling :func:`main` performs the explicit dependency check.
"""

from __future__ import annotations

from typing import Sequence

from spimaging.desktop.dependency import DesktopDependencyError


def main(argv: Sequence[str] | None = None) -> int:
    from spimaging.desktop.application import main as application_main

    return application_main(argv)


__all__ = ["DesktopDependencyError", "main"]
