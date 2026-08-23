"""``python -m spimaging.desktop`` entry point."""

from __future__ import annotations

import sys
from typing import Sequence

from spimaging.desktop import DesktopDependencyError


def main(argv: Sequence[str] | None = None) -> int:
    try:
        from spimaging.desktop.application import main as application_main

        return application_main(argv)
    except DesktopDependencyError as exc:
        print(f"spimaging-desktop: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
