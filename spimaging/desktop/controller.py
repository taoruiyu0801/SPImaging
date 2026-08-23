"""QProcess ownership and schema-v1 event handling for one experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from spimaging.appcore.config import RunConfig
from spimaging.appcore.events import WorkerEvent
from spimaging.appcore.process import CancellationToken, WindowsJob
from spimaging.appcore.storage import RunLayout, atomic_write_text
from spimaging.desktop.dependency import require_pyside6
from spimaging.desktop.models import RunProgressState, TERMINAL_STATUSES


require_pyside6()

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal  # noqa: E402


def normalize_worker_python(executable: str | Path) -> str:
    """Use console ``python.exe`` when the GUI itself runs under ``pythonw.exe``."""

    candidate = Path(executable).expanduser()
    if candidate.name.lower() == "pythonw.exe":
        return str(candidate.with_name("python.exe"))
    return str(candidate)


class WorkerController(QObject):
    """Launch a worker, parse JSONL and own cooperative/forced cancellation."""

    started = Signal(object)
    event_received = Signal(object)
    progress_changed = Signal(object)
    state_changed = Signal(str)
    log_received = Signal(str)
    protocol_warning = Signal(str)
    completed = Signal(str, int)
    launch_failed = Signal(str)

    CANCEL_GRACE_MS = 10_000

    def __init__(self, parent: QObject | None = None, *, python_executable: str | None = None) -> None:
        super().__init__(parent)
        self.python_executable = normalize_worker_python(python_executable or sys.executable)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.started.connect(self._on_started)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self._buffer = ""
        self._config: RunConfig | None = None
        self._progress: RunProgressState | None = None
        self._job: WindowsJob | None = None
        self._cancel_timer = QTimer(self)
        self._cancel_timer.setSingleShot(True)
        self._cancel_timer.timeout.connect(self._force_stop)
        self._cancel_requested = False
        self._finished_emitted = False

    @property
    def config(self) -> RunConfig | None:
        return self._config

    @property
    def progress(self) -> RunProgressState | None:
        return self._progress

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, config: RunConfig) -> None:
        if self.is_running:
            raise RuntimeError("已有任务正在运行")
        run_dir = Path(config.output.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "run.json"
        if config_path.exists():
            existing = RunConfig.load(config_path)
            if existing.run_id != config.run_id:
                raise ValueError("运行目录已属于另一个任务")
        atomic_write_text(config_path, config.to_json())

        self._config = config
        self._progress = RunProgressState(config.run_id)
        self._buffer = ""
        self._cancel_requested = False
        self._finished_emitted = False
        self._cancel_timer.stop()
        self.process.setProgram(self.python_executable)
        self.process.setArguments(["-u", "-m", "spimaging.worker", "--config", str(config_path)])
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("MPLBACKEND", "Agg")
        self.process.setProcessEnvironment(environment)
        self.state_changed.emit("preparing")
        self.process.start()

    def request_cancel(self) -> bool:
        if not self.is_running or self._config is None:
            return False
        if self._cancel_requested:
            return True
        self._cancel_requested = True
        if self._progress is not None:
            self._progress.status = "cancelling"
            self.progress_changed.emit(self._progress)
        self.state_changed.emit("cancelling")
        token = CancellationToken(RunLayout.for_root(Path(self._config.output.run_dir)).cancel_request)
        try:
            token.request()
        except OSError as exc:
            self.protocol_warning.emit(f"无法写入取消请求文件：{exc}")
        self.process.write((json.dumps({"command": "cancel"}) + "\n").encode("utf-8"))
        self._cancel_timer.start(self.CANCEL_GRACE_MS)
        return True

    def _on_started(self) -> None:
        if self._config is None:
            return
        try:
            self._job = WindowsJob()
            assigned = self._job.assign(int(self.process.processId()))
            if os.name == "nt" and not assigned:
                self.protocol_warning.emit("无法将 worker 加入 Windows Job Object；仍会使用进程级取消。")
        except OSError as exc:
            self._job = None
            self.protocol_warning.emit(f"无法创建 Windows Job Object：{exc}；仍会使用进程级取消。")
        self.started.emit(self._config)

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.rstrip("\r"))

    def _consume_line(self, line: str) -> None:
        if not line:
            return
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("事件根节点不是对象")
            event = WorkerEvent.from_dict(raw)
            if self._progress is None:
                raise ValueError("当前没有运行任务")
            self._progress.apply(event)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.log_received.emit(line)
            self.protocol_warning.emit(f"忽略无法解析的 worker 输出：{exc}")
            return
        self.event_received.emit(event)
        self.progress_changed.emit(self._progress)
        if event.type == "log":
            self.log_received.emit(str(event.payload.get("message", "")))
        if event.type == "state":
            self.state_changed.emit(self._progress.status)

    def _on_error(self, process_error: QProcess.ProcessError) -> None:
        if process_error == QProcess.ProcessError.Crashed and self._cancel_requested:
            return
        message = self.process.errorString()
        self.launch_failed.emit(message)
        if process_error == QProcess.ProcessError.FailedToStart and not self._finished_emitted:
            if self._progress is not None:
                self._progress.status = "failed"
                self._progress.error = message
                self.progress_changed.emit(self._progress)
            self._finished_emitted = True
            self.state_changed.emit("failed")
            self.completed.emit("failed", -1)

    def _force_stop(self) -> None:
        if not self.is_running:
            return
        if self._job is not None:
            self._job.terminate(130)
        self.process.kill()
        self.protocol_warning.emit("任务在 10 秒内未结束，已强制终止进程树。")

    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._cancel_timer.stop()
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
        self._buffer = ""
        if self._job is not None:
            self._job.close()
            self._job = None
        status = "interrupted"
        if self._progress is not None:
            status = self._progress.status
            if status not in TERMINAL_STATUSES:
                if self._cancel_requested:
                    status = "cancelled"
                elif exit_code != 0:
                    status = "interrupted"
                else:
                    status = "succeeded"
                self._progress.status = status
                self.progress_changed.emit(self._progress)
        self.state_changed.emit(status)
        if not self._finished_emitted:
            self._finished_emitted = True
            self.completed.emit(status, int(exit_code))

    def dispose(self) -> None:
        """Stop owned work before the controller is discarded."""

        if self.is_running:
            self.request_cancel()


__all__ = ["WorkerController", "normalize_worker_python"]
