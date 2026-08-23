"""Structured JSONL worker events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import threading
from typing import Any, Mapping, TextIO


WORKER_EVENT_SCHEMA_VERSION = 1


class EventType(str, Enum):
    STATE = "state"
    STAGE_STARTED = "stage_started"
    STAGE_PROGRESS = "stage_progress"
    STAGE_COMPLETED = "stage_completed"
    EPOCH = "epoch"
    BATCH = "batch"
    SAMPLE = "sample"
    METRIC = "metric"
    ARTIFACT = "artifact"
    WARNING = "warning"
    LOG = "log"
    ERROR = "error"
    COMPLETED = "completed"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"事件 payload 不是有效 JSON：{exc}") from exc
    return result


@dataclass(frozen=True)
class WorkerEvent:
    schema_version: int
    run_id: str
    seq: int
    timestamp: str
    type: str
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        run_id: str,
        seq: int,
        event_type: EventType | str,
        payload: Mapping[str, Any] | None = None,
    ) -> "WorkerEvent":
        name = event_type.value if isinstance(event_type, EventType) else str(event_type)
        if name not in {item.value for item in EventType}:
            raise ValueError(f"未知 worker 事件类型：{name}")
        if seq < 1:
            raise ValueError("事件序号必须从 1 开始")
        return cls(
            WORKER_EVENT_SCHEMA_VERSION,
            run_id,
            seq,
            _timestamp(),
            name,
            _json_mapping(payload or {}),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkerEvent":
        required = {"schema_version", "run_id", "seq", "timestamp", "type", "payload"}
        if set(data) != required:
            raise ValueError("WorkerEvent 字段不完整或包含未知字段")
        version = int(data["schema_version"])
        if version != WORKER_EVENT_SCHEMA_VERSION:
            raise ValueError(f"不支持的 WorkerEvent schema_version：{version}")
        name = str(data["type"])
        if name not in {item.value for item in EventType}:
            raise ValueError(f"未知 worker 事件类型：{name}")
        return cls(
            version,
            str(data["run_id"]),
            int(data["seq"]),
            str(data["timestamp"]),
            name,
            _json_mapping(data["payload"] if isinstance(data["payload"], Mapping) else {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class EventWriter:
    """Thread-safe event writer that mirrors JSONL to stdout and a run file."""

    def __init__(
        self,
        run_id: str,
        *,
        stream: TextIO | None = None,
        event_path: str | Path | None = None,
    ) -> None:
        self.run_id = run_id
        self.stream = stream
        self.event_path = Path(event_path) if event_path is not None else None
        self._seq = 0
        self._lock = threading.Lock()
        if self.event_path is not None:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event_type: EventType | str,
        payload: Mapping[str, Any] | None = None,
    ) -> WorkerEvent:
        with self._lock:
            self._seq += 1
            event = WorkerEvent.create(self.run_id, self._seq, event_type, payload)
            line = event.to_json() + "\n"
            if self.event_path is not None:
                with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line)
                    handle.flush()
            if self.stream is not None:
                self.stream.write(line)
                self.stream.flush()
            return event

