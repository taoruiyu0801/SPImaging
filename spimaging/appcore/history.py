"""Rebuildable SQLite run-history index."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterable
import uuid

from spimaging.appcore.config import RunConfig
from spimaging.appcore.storage import RUN_STATUSES, ResultManifest, now_iso


@dataclass(frozen=True)
class HistoryRecord:
    run_id: str
    display_name: str
    workflow: str
    status: str
    run_dir: str
    created_at: str
    updated_at: str
    summary: dict


class HistoryStore:
    """SQLite is an index only; authoritative data remains in each run directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.recovered_corrupt_path: Path | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize()
        except sqlite3.DatabaseError:
            self.recovered_corrupt_path = self._preserve_corrupt_database()
            self._initialize()

    def _preserve_corrupt_database(self) -> Path:
        """Move an unreadable index aside; authoritative run directories stay untouched."""

        backup = self.path.with_name(
            f"{self.path.name}.corrupt-{uuid.uuid4().hex}"
        )
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                os.replace(sidecar, Path(str(backup) + suffix))
        if self.path.exists():
            os.replace(self.path, backup)
        return backup

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_updated_idx ON runs(updated_at DESC)"
            )
            connection.commit()

    def upsert(
        self,
        config: RunConfig,
        status: str,
        summary: dict | None = None,
    ) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"未知运行状态：{status}")
        updated_at = now_iso()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, display_name, workflow, status, run_dir,
                    created_at, updated_at, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    workflow=excluded.workflow,
                    status=excluded.status,
                    run_dir=excluded.run_dir,
                    updated_at=excluded.updated_at,
                    summary_json=excluded.summary_json
                """,
                (
                    config.run_id,
                    config.display_name,
                    config.workflow,
                    status,
                    str(Path(config.output.run_dir).expanduser().resolve()),
                    config.created_at,
                    updated_at,
                    json.dumps(summary or {}, ensure_ascii=False, allow_nan=False),
                ),
            )
            connection.commit()

    def list(self, *, limit: int = 100) -> list[HistoryRecord]:
        if limit < 1:
            raise ValueError("history limit 必须大于 0")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            HistoryRecord(
                row["run_id"],
                row["display_name"],
                row["workflow"],
                row["status"],
                row["run_dir"],
                row["created_at"],
                row["updated_at"],
                json.loads(row["summary_json"]),
            )
            for row in rows
        ]

    def mark_interrupted(self) -> int:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status='interrupted', updated_at=?
                WHERE status IN ('preparing', 'running', 'cancelling')
                """,
                (now_iso(),),
            )
            connection.commit()
            return cursor.rowcount

    def rebuild(self, run_roots: Iterable[str | Path]) -> tuple[int, list[str]]:
        imported = 0
        errors: list[str] = []
        for root_value in run_roots:
            root = Path(root_value).expanduser().resolve()
            candidates = [root] if (root / "run.json").is_file() else sorted(root.glob("*/"))
            for candidate in candidates:
                config_path = candidate / "run.json"
                if not config_path.is_file():
                    continue
                try:
                    config = RunConfig.load(config_path)
                    result_path = candidate / "result_manifest.json"
                    status = "interrupted"
                    summary: dict = {}
                    if result_path.is_file():
                        raw = json.loads(result_path.read_text(encoding="utf-8"))
                        result = ResultManifest.from_dict(raw)
                        status = result.status
                        summary = result.metrics
                    self.upsert(config, status, summary)
                    imported += 1
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{candidate}: {exc}")
        return imported, errors
