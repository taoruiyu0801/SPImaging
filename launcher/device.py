"""Small, dependency-free NVIDIA capability probe for runtime selection."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from .manifest import NvidiaCapability


def probe_nvidia(timeout_seconds: int = 8) -> NvidiaCapability:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        common = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "NVIDIA Corporation",
            "NVSMI",
            "nvidia-smi.exe",
        )
        executable = common if os.path.isfile(common) else None
    if executable is None:
        return NvidiaCapability(False, reason="未找到 nvidia-smi，可能没有安装兼容的 NVIDIA 驱动")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return NvidiaCapability(False, reason=f"NVIDIA 驱动检测失败：{error}")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "未知错误").strip().splitlines()[0]
        return NvidiaCapability(False, reason=f"nvidia-smi 检测失败：{detail}")
    first = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    match = re.search(r"\d+(?:\.\d+){0,3}", first)
    if match is None:
        return NvidiaCapability(False, reason="nvidia-smi 未返回可识别的驱动版本")
    return NvidiaCapability(True, driver_version=match.group(0), reason="检测到 NVIDIA GPU")
