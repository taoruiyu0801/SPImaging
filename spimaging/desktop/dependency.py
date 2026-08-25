"""Explicit optional dependency boundary for the source desktop."""

from __future__ import annotations


class DesktopDependencyError(ImportError):
    """Raised when the optional PySide6 desktop runtime is unavailable."""


def require_pyside6() -> None:
    try:
        import PySide6  # noqa: F401
    except ImportError as exc:
        raise DesktopDependencyError(
            "SPImaging 图形工作台需要 PySide6。请安装桌面版依赖，或使用现有命令行入口。"
        ) from exc


__all__ = ["DesktopDependencyError", "require_pyside6"]
