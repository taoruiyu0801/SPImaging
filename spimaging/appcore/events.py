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
        self._needs_separator = False
        self._lock = threading.Lock()
        if self.event_path is not None:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            self._seq, self._needs_separator = self._load_existing_sequence()

    def _load_existing_sequence(self) -> tuple[int, bool]:
        """Continue one run's JSONL sequence without rewriting prior events."""

        assert self.event_path is not None
        try:
            size = self.event_path.stat().st_size
        except FileNotFoundError:
            return 0, False
        if size == 0:
            return 0, False

        with self.event_path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_separator = handle.read(1) not in {b"\n", b"\r"}
            handle.seek(0)
            sequence = 0
            offset = 0
            truncate_at: int | None = None
            for line_number, raw_line in enumerate(handle, 1):
                line_start = offset
                offset += len(raw_line)
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    decoded = stripped.decode("utf-8")
                    data = json.loads(decoded)
                    if not isinstance(data, Mapping):
                        raise ValueError("事件必须是 JSON 对象")
                    event = WorkerEvent.from_dict(data)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    # A hard process termination can leave only the final JSON
                    # object partially written. Drop only those incomplete tail
                    # bytes so this and every later restart remain readable.
                    if offset == size and needs_separator:
                        truncate_at = line_start
                        needs_separator = False
                        break
                    raise ValueError(
                        f"现有事件文件第 {line_number} 行无效：{exc}"
                    ) from exc
                if event.run_id != self.run_id:
                    raise ValueError("现有事件文件属于另一个 run_id")
                if event.seq != sequence + 1:
                    raise ValueError("现有事件序号不连续")
                sequence = event.seq
        if truncate_at is not None:
            with self.event_path.open("r+b") as handle:
                handle.truncate(truncate_at)
        return sequence, needs_separator

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
                    if self._needs_separator:
                        handle.write("\n")
                        self._needs_separator = False
                    handle.write(line)
                    handle.flush()
            if self.stream is not None:
                self.stream.write(line)
                self.stream.flush()
            return event
