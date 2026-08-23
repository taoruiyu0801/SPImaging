"""Atomic run-directory and result-manifest storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

from spimaging.appcore.config import RunConfig


RESULT_MANIFEST_SCHEMA_VERSION = 1
RUN_STATUSES = {
    "preparing",
    "running",
    "cancelling",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def safe_relative_path(value: str | Path) -> str:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or not raw or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"产物路径必须是安全的相对路径：{value}")
    if path.parts and ":" in path.parts[0]:
        raise ValueError(f"产物路径不能包含驱动器：{value}")
    return path.as_posix()


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    path: str
    kind: str
    sample_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRecord":
        path = safe_relative_path(str(data.get("path", "")))
        sample_index = data.get("sample_index")
        if sample_index is not None and int(sample_index) < 0:
            raise ValueError("artifact.sample_index 不能为负数")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("artifact.metadata 必须是对象")
        return cls(
            name=str(data.get("name", "")),
            path=path,
            kind=str(data.get("kind", "file")),
            sample_index=None if sample_index is None else int(sample_index),
            metadata=dict(metadata),
        )


@dataclass
class ResultManifest:
    schema_version: int
    run_id: str
    status: str
    workflow: str
    started_at: str
    completed_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None

    @classmethod
    def new(cls, config: RunConfig) -> "ResultManifest":
        return cls(
            RESULT_MANIFEST_SCHEMA_VERSION,
            config.run_id,
            "preparing",
            config.workflow,
            now_iso(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResultManifest":
        version = int(data.get("schema_version", 0))
        if version != RESULT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"不支持的 ResultManifest schema_version：{version}")
        status = str(data.get("status", ""))
        if status not in RUN_STATUSES:
            raise ValueError(f"未知运行状态：{status}")
        artifacts = data.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError("result artifacts 必须是列表")
        metrics = data.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("result metrics 必须是对象")
        warnings = data.get("warnings", [])
        if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
            raise ValueError("result warnings 必须是文本列表")
        error = data.get("error")
        if error is not None and not isinstance(error, Mapping):
            raise ValueError("result error 必须是对象或 null")
        return cls(
            version,
            str(data.get("run_id", "")),
            status,
            str(data.get("workflow", "")),
            str(data.get("started_at", "")),
            None if data.get("completed_at") is None else str(data["completed_at"]),
            dict(metrics),
            [ArtifactRecord.from_dict(item) for item in artifacts if isinstance(item, Mapping)],
            list(warnings),
            None if error is None else dict(error),
        )

    def add_artifact(
        self,
        name: str,
        path: str | Path,
        kind: str,
        *,
        sample_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            name,
            safe_relative_path(path),
            kind,
            sample_index,
            dict(metadata or {}),
        )
        self.artifacts.append(record)
        return record

    def set_status(self, status: str, *, error: Mapping[str, Any] | None = None) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"未知运行状态：{status}")
        self.status = status
        self.error = None if error is None else dict(error)
        if status in {"succeeded", "failed", "cancelled", "interrupted"}:
            self.completed_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunLayout:
    root: Path
    config: Path
    events: Path
    result: Path
    log: Path
    cancel_request: Path
    artifacts: Path
    checkpoints: Path
    metrics: Path
    gallery: Path
    training: Path

    @classmethod
    def for_root(cls, root: Path) -> "RunLayout":
        return cls(
            root=root,
            config=root / "run.json",
            events=root / "events.jsonl",
            result=root / "result_manifest.json",
            log=root / "logs" / "worker.log",
            cancel_request=root / "cancel.request",
            artifacts=root / "artifacts",
            checkpoints=root / "checkpoints",
            metrics=root / "metrics",
            gallery=root / "gallery",
            training=root / "training",
        )


class RunStorage:
    """Own one non-symlink run directory and its atomic manifests."""

    def __init__(self, config: RunConfig) -> None:
        root = Path(config.output.run_dir).expanduser().resolve()
        self.config = config
        self.layout = RunLayout.for_root(root)

    def prepare(self) -> ResultManifest:
        root = self.layout.root
        if root.is_symlink():
            raise ValueError(f"运行目录不能是符号链接：{root}")
        if root.exists() and not root.is_dir():
            raise ValueError(f"运行路径不是目录：{root}")
        root.mkdir(parents=True, exist_ok=True)
        if self.layout.config.exists():
            existing = RunConfig.load(self.layout.config)
            if existing.run_id != self.config.run_id:
                raise ValueError(f"运行目录已属于另一个任务：{root}")
        for directory in (
            self.layout.log.parent,
            self.layout.artifacts,
            self.layout.checkpoints,
            self.layout.metrics,
            self.layout.gallery,
            self.layout.training,
        ):
            if directory.is_symlink():
                raise ValueError(f"运行子目录不能是符号链接：{directory}")
            directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.layout.config, self.config.to_json())
        manifest = ResultManifest.new(self.config)
        self.write_result(manifest)
        return manifest

    def write_result(self, manifest: ResultManifest) -> None:
        if manifest.run_id != self.config.run_id:
            raise ValueError("result manifest run_id 与配置不一致")
        atomic_write_json(self.layout.result, manifest.to_dict())

    def relative_to_root(self, path: str | Path) -> str:
        candidate = Path(path).resolve()
        try:
            relative = candidate.relative_to(self.layout.root)
        except ValueError as exc:
            raise ValueError(f"产物位于运行目录之外：{candidate}") from exc
        return safe_relative_path(relative)

    def load_result(self) -> ResultManifest:
        try:
            data = json.loads(self.layout.result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取结果清单：{exc}") from exc
        if not isinstance(data, Mapping):
            raise ValueError("结果清单根节点必须是对象")
        result = ResultManifest.from_dict(data)
        if result.run_id != self.config.run_id:
            raise ValueError("结果清单 run_id 与配置不一致")
        return result

