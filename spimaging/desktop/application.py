"""Source and packaged desktop application bootstrap."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from spimaging import PRODUCT_VERSION, __version__, product_display_name
from spimaging.desktop.dependency import require_pyside6


def create_application(argv: Sequence[str] | None = None):
    require_pyside6()
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    from spimaging.desktop.i18n import install_translator
    from spimaging.desktop.models import ApplicationPaths, SettingsStore
    from spimaging.desktop.style import APP_STYLESHEET

    application = QApplication.instance() or QApplication(list(sys.argv if argv is None else argv))
    QCoreApplication.setOrganizationName("SPImaging")
    QCoreApplication.setApplicationName("SPImaging")
    QCoreApplication.setApplicationVersion(__version__)
    application.setApplicationDisplayName(product_display_name())
    application.setStyleSheet(APP_STYLESHEET)
    paths = ApplicationPaths.default()
    settings = SettingsStore(paths.settings_file, paths).load()
    install_translator(application, settings.locale)
    return application


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spimaging-desktop",
        description="SPImaging PySide6 单光子成像实验工作台",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"SPImaging {PRODUCT_VERSION} (build {__version__})",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="离屏构造主窗口后立即退出，用于安装健康检查。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    options, qt_arguments = build_parser().parse_known_args(arguments)
    application = create_application([sys.argv[0], *qt_arguments])
    from spimaging.desktop.window import MainWindow

    window = MainWindow()
    if options.smoke_test:
        window.show()
        application.processEvents()
        window.close()
        application.processEvents()
        return 0
    window.show()
    return int(application.exec())


__all__ = ["build_parser", "create_application", "main"]
