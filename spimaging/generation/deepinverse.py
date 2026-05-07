"""DeepInverse import compatibility helpers."""

from __future__ import annotations


def import_deepinv():
    """Import DeepInverse after loading Pillow's image extension on Windows."""
    from PIL import Image  # noqa: F401
    import deepinv as dinv

    return dinv
