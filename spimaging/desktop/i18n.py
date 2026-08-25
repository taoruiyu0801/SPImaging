"""Qt translation helpers.

Chinese is the source language for the beta.  Every visible string still goes
through ``QCoreApplication.translate`` so an English ``.qm`` catalogue can be
added without touching workflow code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def tr(context: str, text: str) -> str:
    try:
        from PySide6.QtCore import QCoreApplication
    except ImportError:
        return text
    return QCoreApplication.translate(context, text)


def install_translator(application, locale: str, search_roots: Iterable[str | Path] = ()):
    """Load the first matching translation catalogue, if one is bundled."""

    from PySide6.QtCore import QLocale, QTranslator

    candidates = [Path(__file__).with_name("translations")]
    candidates.extend(Path(item) for item in search_roots)
    names = (f"spimaging_{locale}.qm", f"spimaging_{QLocale(locale).name()}.qm")
    for root in candidates:
        for name in names:
            path = root / name
            if not path.is_file():
                continue
            translator = QTranslator(application)
            if translator.load(str(path)):
                application.installTranslator(translator)
                # QApplication does not take Python ownership of QTranslator.
                application._spimaging_translator = translator
                return translator
    return None


__all__ = ["install_translator", "tr"]
