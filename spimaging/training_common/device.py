"""Explicit, explainable PyTorch device selection."""

from __future__ import annotations

from dataclasses import dataclass
import os


DEVICE_CHOICES = ("auto", "cuda", "cpu")


@dataclass(frozen=True)
class DeviceSelection:
    device: object
    requested: str
    gpu_index: int
    reason: str
    fallback: bool


def select_torch_device(mode: str = "auto", gpu_index: int = 0) -> DeviceSelection:
    """Select and smoke-test a device, falling back to CPU with a reason."""

    import torch

    requested = str(mode).lower()
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"device mode must be one of: {', '.join(DEVICE_CHOICES)}")
    if isinstance(gpu_index, bool) or int(gpu_index) < 0:
        raise ValueError("gpu_index must be a nonnegative integer")
    gpu_index = int(gpu_index)

    if requested == "cpu":
        return DeviceSelection(
            device=torch.device("cpu"),
            requested=requested,
            gpu_index=gpu_index,
            reason="CPU was explicitly selected.",
            fallback=False,
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # driver probing can fail outside normal torch errors
        return DeviceSelection(
            device=torch.device("cpu"),
            requested=requested,
            gpu_index=gpu_index,
            reason=f"CUDA probe failed ({exc}); using CPU.",
            fallback=True,
        )

    if not cuda_available:
        return DeviceSelection(
            device=torch.device("cpu"),
            requested=requested,
            gpu_index=gpu_index,
            reason="CUDA is not available in this runtime or driver; using CPU.",
            fallback=True,
        )

    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        return DeviceSelection(
            device=torch.device("cpu"),
            requested=requested,
            gpu_index=gpu_index,
            reason=f"CUDA device enumeration failed ({exc}); using CPU.",
            fallback=True,
        )
    logical_gpu_index = gpu_index
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if len(visible_tokens) == 1 and visible_tokens[0] == str(gpu_index):
        logical_gpu_index = 0
    if logical_gpu_index >= device_count:
        return DeviceSelection(
            device=torch.device("cpu"),
            requested=requested,
            gpu_index=gpu_index,
            reason=(
                f"CUDA device index {gpu_index} is unavailable "
                f"({device_count} device(s) detected); using CPU."
            ),
            fallback=True,
        )

    candidate = torch.device(f"cuda:{logical_gpu_index}")
    try:
        torch.empty(1, device=candidate)
        name = torch.cuda.get_device_name(logical_gpu_index)
    except Exception as exc:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return DeviceSelection(
            device=torch.device("cpu"),
            requested=requested,
            gpu_index=gpu_index,
            reason=f"CUDA device {gpu_index} self-test failed ({exc}); using CPU.",
            fallback=True,
        )

    return DeviceSelection(
        device=candidate,
        requested=requested,
        gpu_index=gpu_index,
        reason=f"Using CUDA device {gpu_index}: {name}.",
        fallback=False,
    )


def get_torch_device(
    prefer_gpu: bool = True,
    *,
    mode: str | None = None,
    gpu_index: int = 0,
    return_selection: bool = False,
):
    """Backward-compatible device helper with optional explicit selection."""

    selected_mode = mode if mode is not None else ("auto" if prefer_gpu else "cpu")
    selection = select_torch_device(selected_mode, gpu_index)
    return selection if return_selection else selection.device
