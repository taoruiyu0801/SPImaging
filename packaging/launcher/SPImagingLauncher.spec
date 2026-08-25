# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller input for the dependency-free bootstrap launcher."""

from pathlib import Path

repo = Path(SPECPATH).resolve().parents[1]

analysis = Analysis(
    [str(repo / "packaging" / "launcher" / "entrypoint.py")],
    pathex=[str(repo)],
    binaries=[],
    datas=[],
    hiddenimports=["tkinter", "tkinter.ttk", "tkinter.messagebox"],
    excludes=[
        "numpy",
        "torch",
        "PySide6",
        "matplotlib",
        "h5py",
        "cv2",
        "skimage",
    ],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="SPImaging",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch="x86_64",
    version=str(repo / "packaging" / "launcher" / "version_info.txt"),
)
