"""Structured algorithm events and cooperative cancellation hooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
from typing import Any


EVENT_PREFIX = "SPIMAGING_EVENT "
STRUCTURED_EVENTS_ENV = "SPIMAGING_STRUCTURED_EVENTS"
CANCEL_FILE_ENV = "SPIMAGING_CANCEL_FILE"

EventCallback = Callable[[Mapping[str, Any]], None]
CancelCheck = Callable[[], bool]


class CancellationRequested(RuntimeError):
    """Signals a cooperative stop at an algorithm-safe boundary."""

    def __init__(
        self,
        message: str = "cancellation requested",
        *,
        phase: str | None = None,
        epoch: int | None = None,
        next_batch: int = 0,
        global_step: int = 0,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.epoch = epoch
        self.next_batch = next_batch
        self.global_step = global_step


def _json_default(value: object):
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


def structured_events_enabled() -> bool:
    return os.environ.get(STRUCTURED_EVENTS_ENV, "").strip() == "1"


def emit_event(
    event_type: str,
    *,
    callback: EventCallback | None = None,
    **payload: object,
) -> dict[str, object]:
    """Deliver a callback event and optionally mirror it as prefixed JSONL."""

    event: dict[str, object] = {"type": str(event_type), **payload}
    if callback is not None:
        callback(event)
    if structured_events_enabled():
        print(
            EVENT_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=_json_default),
            flush=True,
        )
    return event


def cancellation_requested(cancel_check: CancelCheck | None = None) -> bool:
    if cancel_check is not None and cancel_check():
        return True
    cancel_path = os.environ.get(CANCEL_FILE_ENV)
    return bool(cancel_path and Path(cancel_path).is_file())


def raise_if_cancelled(
    cancel_check: CancelCheck | None = None,
    *,
    phase: str | None = None,
    epoch: int | None = None,
    next_batch: int = 0,
    global_step: int = 0,
) -> None:
    if cancellation_requested(cancel_check):
        raise CancellationRequested(
            phase=phase,
            epoch=epoch,
            next_batch=next_batch,
            global_step=global_step,
        )
