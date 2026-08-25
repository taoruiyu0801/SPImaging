"""Run one configured SPImaging experiment and emit structured JSONL events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, TextIO

from spimaging.appcore.config import RunConfig
from spimaging.appcore.events import EventType, EventWriter
from spimaging.appcore.history import HistoryStore
from spimaging.appcore.process import CancellationToken, WindowsJob
from spimaging.appcore.specs import (
    GENERATION_COMMON_PARAMETERS,
    RECONSTRUCTION_ALGORITHMS,
    SIMULATION_ALGORITHMS,
)
from spimaging.appcore.storage import ResultManifest, RunStorage
from spimaging.generation.recovery import partial_directory_for


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHILD_EVENT_PREFIX = "SPIMAGING_EVENT "
TERMINATION_GRACE_SECONDS = 10.0
TERMINAL_RESERVE_BYTES = 64 * 1024


class WorkerError(RuntimeError):
    """Expected run failure with a user-facing message."""


class WorkerCancelled(WorkerError):
    """The user cancelled the configured run."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spimaging-worker",
        description="Execute one versioned SPImaging RunConfig and stream JSONL events.",
    )
    parser.add_argument("--config", required=True, help="Path to a schema-v1 run.json file.")
    return parser


def _flag_value(arguments: list[str], flag: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            arguments.append(flag)
        return
    arguments.extend((flag, str(value)))


def _effective_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Remove GUI sentinels used for optional positive CLI parameters."""

    return {
        key: value
        for key, value in values.items()
        if value is not None
        and not (key in {"max_samples", "early_stopping_patience"} and value == 0)
    }


def _append_parameter_arguments(
    arguments: list[str],
    values: Mapping[str, Any],
    specs: Iterable[Any],
) -> None:
    known = {parameter.name: parameter for parameter in specs}
    for name, value in _effective_values(values).items():
        parameter = known.get(name)
        if parameter is None:
            continue
        if not parameter.is_visible(values):
            continue
        _flag_value(arguments, parameter.cli_flag or f"--{name}", value)


def _module_command(module: str, arguments: Iterable[str]) -> list[str]:
    return [sys.executable, "-u", "-m", module, *list(arguments)]


def _list_samples(dataset_dir: Path) -> list[Path]:
    samples = sorted(dataset_dir.glob("sample_*.npz"))
    if not samples:
        samples = sorted(dataset_dir.glob("*.npz"))
    return samples


def _public_demo_dataset() -> Path:
    candidates = (
        REPOSITORY_ROOT / "public_demo" / "dataset",
        REPOSITORY_ROOT / "public_demo" / "data",
        REPOSITORY_ROOT / "public_demo",
    )
    for candidate in candidates:
        if candidate.is_dir() and _list_samples(candidate):
            return candidate.resolve()
    raise WorkerError("公开合成演示数据尚未安装；请修复应用资源或选择数据集。")


def _history_path(config: RunConfig) -> Path:
    if config.output.history_db:
        return Path(config.output.history_db).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    root = Path(local) if local else Path.home() / ".spimaging"
    return root / "SPImaging" / "history.sqlite3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_generation_is_valid(output_dir: Path) -> bool:
    """Only reuse a published generation whose recorded samples still match."""

    manifest_path = output_dir / "generation_manifest.json"
    try:
        if manifest_path.stat().st_size > 8 * 1024 * 1024:
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, Mapping):
        return False
    completed = manifest.get("completed")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        return False
    if not isinstance(completed, list) or not completed:
        return False
    for item in completed:
        if not isinstance(item, Mapping):
            return False
        name = item.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            return False
        sample = output_dir / name
        try:
            if sample.stat().st_size != item.get("bytes") or _sha256(sample) != item.get("sha256"):
                return False
        except OSError:
            return False
    return True


class WorkerRuntime:
    """Stateful adapter from RunConfig to existing isolated CLI modules."""

    def __init__(self, config: RunConfig, *, stream: TextIO | None = None) -> None:
        self.config = config
        self.storage = RunStorage(config)
        self.layout = self.storage.layout
        self.cancel = CancellationToken(self.layout.cancel_request)
        self.events = EventWriter(config.run_id, stream=stream, event_path=self.layout.events)
        self.manifest: ResultManifest | None = None
        self.history = HistoryStore(_history_path(config))
        self._control_thread: threading.Thread | None = None
        self._terminal_reserve = self.layout.root / ".terminal-reserve"
        self._storage_prepared = False

    def start_control_reader(self, stream: TextIO | None = None) -> None:
        source = stream if stream is not None else sys.stdin

        def read_commands() -> None:
            try:
                for line in source:
                    try:
                        command = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(command, Mapping) and command.get("command") == "cancel":
                        self.cancel.request()
                        return
            except (OSError, ValueError):
                return

        self._control_thread = threading.Thread(
            target=read_commands,
            name="spimaging-worker-control",
            daemon=True,
        )
        self._control_thread.start()

    def _set_status(self, status: str, **payload: Any) -> None:
        assert self.manifest is not None
        self.manifest.set_status(status, error=payload.get("error"))
        self.storage.write_result(self.manifest)
        self.history.upsert(self.config, status, self.manifest.metrics)
        self.events.emit(EventType.STATE, {"status": status, **payload})

    def _create_terminal_reserve(self) -> None:
        """Reserve enough allocated bytes for a compact terminal record."""

        reserve = self._terminal_reserve
        try:
            if reserve.is_symlink():
                raise OSError("terminal reserve cannot be a symbolic link")
            if reserve.exists():
                if not reserve.is_file():
                    raise OSError("terminal reserve path is not a file")
                reserve.unlink()
            with reserve.open("xb") as handle:
                handle.write(b"\0" * TERMINAL_RESERVE_BYTES)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            # Reserving space is an optimization. Failure here must not stop a
            # run that can otherwise proceed, and terminalization still tries
            # every normal and compact persistence path.
            try:
                if reserve.is_file() and not reserve.is_symlink():
                    reserve.unlink()
            except OSError:
                pass

    def _release_terminal_reserve(self) -> None:
        try:
            if self._terminal_reserve.is_file() and not self._terminal_reserve.is_symlink():
                self._terminal_reserve.unlink()
        except OSError:
            pass

    def _confined_terminal_path(self, path: Path) -> Path:
        """Accept only fixed direct children of this run directory."""

        root = self.layout.root.resolve()
        candidate = path.absolute()
        if candidate.parent.resolve() != root:
            raise ValueError("terminal fallback path escaped the run directory")
        if candidate.name not in {"result_manifest.json", "events.jsonl"}:
            raise ValueError("terminal fallback path is not an authoritative run file")
        if candidate.is_symlink():
            raise ValueError("terminal fallback refuses symbolic-link targets")
        self._assert_run_directory_ownership()
        return candidate

    def _assert_run_directory_ownership(self) -> None:
        if self._storage_prepared:
            return
        try:
            existing = RunConfig.load(self.layout.config)
        except ValueError as exc:
            raise ValueError(
                "terminal fallback cannot verify ownership of the run directory"
            ) from exc
        configured_root = Path(existing.output.run_dir).expanduser().resolve()
        if existing.run_id != self.config.run_id or configured_root != self.layout.root.resolve():
            raise ValueError("terminal fallback run directory belongs to another task")

    def _write_compact_terminal_manifest(
        self,
        status: str,
        error: Mapping[str, Any] | None,
        secondary_errors: list[str],
    ) -> None:
        """Write an authoritative manifest, compacting only when space requires it."""

        path = self._confined_terminal_path(self.layout.result)
        emergency_warnings = [
            "终态使用紧急精简清单写入。",
            *(message[:1000] for message in secondary_errors[-4:]),
        ]
        fallback = self.manifest
        encoded: str | None = None
        if fallback is not None:
            for warning in emergency_warnings:
                if warning not in fallback.warnings:
                    fallback.warnings.append(warning)
            try:
                candidate = json.dumps(
                    fallback.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ) + "\n"
                if len(candidate.encode("utf-8")) <= TERMINAL_RESERVE_BYTES * 3 // 4:
                    encoded = candidate
            except (TypeError, ValueError):
                encoded = None
        if encoded is None:
            compact_error = None
            if error is not None:
                compact_error = {
                    "category": str(error.get("category", "WorkerError"))[:200],
                    "message": str(error.get("message", ""))[:4000],
                }
                secondary = error.get("secondary_errors")
                if isinstance(secondary, list):
                    compact_error["secondary_errors"] = [
                        str(item)[:1000] for item in secondary[-4:]
                    ]
            fallback = ResultManifest.new(self.config)
            if self.manifest is not None:
                fallback.started_at = self.manifest.started_at
            fallback.warnings = emergency_warnings
            fallback.set_status(status, error=compact_error)
            encoded = json.dumps(
                fallback.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ) + "\n"
        temporary = path.with_name(
            f".{path.name}.terminal-{os.getpid()}-{time.time_ns()}.tmp"
        )
        if temporary.parent != path.parent:
            raise ValueError("terminal fallback temporary path escaped the run directory")
        try:
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            except Exception as atomic_exc:
                # Last resort for a full filesystem: truncating the already
                # allocated authoritative file can succeed when creating or
                # renaming another directory entry cannot. This path is used
                # only after the atomic attempt failed.
                try:
                    with path.open("w", encoding="utf-8", newline="\n") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                except Exception as direct_exc:
                    raise OSError(
                        "atomic and in-place terminal manifest writes both failed: "
                        f"{atomic_exc}; {direct_exc}"
                    ) from direct_exc
        finally:
            try:
                if temporary.exists() and not temporary.is_symlink():
                    temporary.unlink()
            except OSError:
                pass
        self.manifest = fallback

    @staticmethod
    def _secondary_error(stage: str, exc: Exception) -> str:
        return f"{stage}: {type(exc).__name__}: {exc}"

    def _emit_terminal_event_best_effort(
        self,
        event_type: EventType,
        payload: Mapping[str, Any],
        secondary_errors: list[str],
    ) -> None:
        try:
            self.events.emit(event_type, payload)
            return
        except Exception as exc:
            secondary_errors.append(self._secondary_error(f"emit {event_type.value}", exc))
        try:
            event_path = self._confined_terminal_path(self.layout.events)
            EventWriter(self.config.run_id, event_path=event_path).emit(event_type, payload)
        except Exception as exc:
            secondary_errors.append(
                self._secondary_error(f"fallback emit {event_type.value}", exc)
            )

    def _finalize_terminal(
        self,
        status: str,
        *,
        error: Mapping[str, Any] | None = None,
        message: str | None = None,
        collect_results: bool = True,
    ) -> None:
        """Persist terminal state without allowing cleanup failures to mask cause."""

        self._release_terminal_reserve()
        secondary_errors: list[str] = []
        if self.manifest is None:
            self.manifest = ResultManifest.new(self.config)

        try:
            self._assert_run_directory_ownership()
            owns_run_directory = True
        except Exception as exc:
            owns_run_directory = False
            secondary_errors.append(self._secondary_error("verify run ownership", exc))

        if collect_results and owns_run_directory:
            try:
                self._collect_results()
            except Exception as exc:
                secondary_errors.append(self._secondary_error("collect_results", exc))

        terminal_error = None if error is None else dict(error)
        if terminal_error is not None and secondary_errors:
            terminal_error["secondary_errors"] = list(secondary_errors)
        self.manifest.warnings.extend(
            f"终态清理警告：{item}" for item in secondary_errors
        )
        try:
            self.manifest.set_status(status, error=terminal_error)
        except Exception as exc:
            secondary_errors.append(self._secondary_error("set terminal status", exc))
            self.manifest = ResultManifest.new(self.config)
            self.manifest.set_status(status, error=terminal_error)

        if not owns_run_directory:
            return

        try:
            self.storage.write_result(self.manifest)
        except Exception as exc:
            secondary_errors.append(self._secondary_error("write terminal manifest", exc))
            try:
                self._write_compact_terminal_manifest(status, terminal_error, secondary_errors)
            except Exception as fallback_exc:
                secondary_errors.append(
                    self._secondary_error("fallback terminal manifest", fallback_exc)
                )

        try:
            self.history.upsert(self.config, status, self.manifest.metrics)
        except Exception as exc:
            secondary_errors.append(self._secondary_error("write terminal history", exc))

        state_payload: dict[str, Any] = {"status": status}
        if terminal_error is not None:
            state_payload["error"] = terminal_error
        if message:
            state_payload["message"] = message
        self._emit_terminal_event_best_effort(
            EventType.STATE,
            state_payload,
            secondary_errors,
        )
        if status == "failed" and terminal_error is not None:
            self._emit_terminal_event_best_effort(
                EventType.ERROR,
                terminal_error,
                secondary_errors,
            )
        completed_payload: dict[str, Any] = {"status": status}
        if status == "succeeded":
            completed_payload["result_manifest"] = str(self.layout.result)
        if message:
            completed_payload["message"] = message
        self._emit_terminal_event_best_effort(
            EventType.COMPLETED,
            completed_payload,
            secondary_errors,
        )

    def _child_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "MPLBACKEND": "Agg",
                "SPIMAGING_STRUCTURED_EVENTS": "1",
                "SPIMAGING_CANCEL_FILE": str(self.layout.cancel_request),
                "SPIMAGING_RUN_ID": self.config.run_id,
                "SPIMAGING_DEVICE": self.config.compute.preference,
                "SPIMAGING_GPU_INDEX": str(self.config.compute.gpu_index),
            }
        )
        if self.config.compute.preference == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        elif self.config.compute.preference == "cuda":
            environment["CUDA_VISIBLE_DEVICES"] = str(self.config.compute.gpu_index)
        prior_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(REPOSITORY_ROOT), prior_pythonpath) if item
        )
        return environment

    def _emit_child_event(self, raw: str, stage: str) -> bool:
        if not raw.startswith(CHILD_EVENT_PREFIX):
            return False
        try:
            payload = json.loads(raw[len(CHILD_EVENT_PREFIX):])
        except json.JSONDecodeError:
            self.events.emit(EventType.WARNING, {"stage": stage, "message": "子进程事件 JSON 无效"})
            return True
        if not isinstance(payload, Mapping):
            return True
        event_name = str(payload.get("type", "stage_progress"))
        event_payload = {key: value for key, value in payload.items() if key != "type"}
        event_payload.setdefault("stage", stage)
        known = {event.value for event in EventType}
        self.events.emit(event_name if event_name in known else EventType.STAGE_PROGRESS, event_payload)
        return True

    def run_command(self, stage: str, command: list[str]) -> None:
        if self.cancel.requested:
            raise WorkerCancelled("任务已取消")
        self.events.emit(EventType.STAGE_STARTED, {"stage": stage, "command": command[2:]})
        self.layout.log.parent.mkdir(parents=True, exist_ok=True)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=self._child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        job = WindowsJob()
        job.assign(process.pid)
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True, name=f"{stage}-output")
        reader.start()
        cancel_started: float | None = None
        stream_ended = False
        try:
            with self.layout.log.open("a", encoding="utf-8", newline="\n") as log_file:
                log_file.write(f"\n[{stage}] {subprocess.list2cmdline(command)}\n")
                while process.poll() is None or not stream_ended:
                    try:
                        line = output_queue.get(timeout=0.1)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        stream_ended = True
                    elif line:
                        log_file.write(line)
                        log_file.flush()
                        stripped = line.rstrip("\r\n")
                        if not self._emit_child_event(stripped, stage):
                            self.events.emit(EventType.LOG, {"stage": stage, "message": stripped})

                    if self.cancel.requested:
                        if cancel_started is None:
                            cancel_started = time.monotonic()
                            self.events.emit(EventType.STATE, {"status": "cancelling", "stage": stage})
                        if time.monotonic() - cancel_started >= TERMINATION_GRACE_SECONDS:
                            job.terminate(130)
                            if process.poll() is None:
                                process.terminate()
                    if stream_ended and process.poll() is not None:
                        break
            returncode = process.wait()
        finally:
            job.close()
        if self.cancel.requested:
            raise WorkerCancelled("任务已由用户取消")
        if returncode != 0:
            raise WorkerError(f"阶段 {stage} 失败，退出码 {returncode}")
        self.events.emit(EventType.STAGE_COMPLETED, {"stage": stage})

    def _generation_command(self, output_dir: Path) -> list[str]:
        config = self.config.generation
        arguments = [
            "--dataset_mode", config.dataset_mode,
            "--surface_model", config.surface_model,
            "--output_dir", str(output_dir),
        ]
        source_flags = {
            "labeled": "--nyu_mat",
            "raw": "--raw_root",
            "middlebury": "--middlebury_root",
        }
        assert self.config.input.source_path is not None
        arguments.extend((source_flags[config.dataset_mode], self.config.input.source_path))
        algorithm = SIMULATION_ALGORITHMS[config.surface_model]
        values = {parameter.name: parameter.default for parameter in GENERATION_COMMON_PARAMETERS}
        values.update(algorithm.parameter_defaults())
        values.update(config.parameters)
        _append_parameter_arguments(
            arguments,
            values,
            (*GENERATION_COMMON_PARAMETERS, *algorithm.parameters),
        )
        partial_exists = partial_directory_for(output_dir).is_dir()
        if config.resume and partial_exists:
            arguments.append("--resume")
        elif not config.resume and (partial_exists or output_dir.exists()):
            arguments.append("--overwrite")
        return _module_command("spimaging.generation.pipeline", arguments)

    def _verify_command(self, dataset_dir: Path, sample_index: int = 0) -> list[str]:
        return _module_command(
            "spimaging.testing.verify",
            (
                "--dataset_dir", str(dataset_dir),
                "--index", str(sample_index),
                "--output_fig", str(self.layout.gallery / f"input_{sample_index:05d}.png"),
                "--overwrite",
            ),
        )

    def _training_command(self, dataset_dir: Path) -> list[str]:
        training = self.config.training
        algorithm = RECONSTRUCTION_ALGORITHMS[training.model]
        module = (
            "spimaging.self_supervised_training.train"
            if algorithm.method_family == "self_supervised_spisr"
            else "spimaging.supervised_training.train"
        )
        arguments: list[str] = []
        dataset_paths = self.config.input.dataset_paths or (str(dataset_dir),)
        for path in dataset_paths:
            arguments.extend(("--dataset_dir", str(path)))
        arguments.extend(("--output_dir", str(self.layout.checkpoints), "--overwrite", "--model", training.model))
        values = training.resolved_parameters()
        from spimaging.appcore.specs import TRAINING_COMMON_PARAMETERS

        _append_parameter_arguments(
            arguments,
            values,
            (*TRAINING_COMMON_PARAMETERS, *algorithm.parameters),
        )
        if training.resume_checkpoint:
            arguments.extend(("--resume_checkpoint", training.resume_checkpoint))
        return _module_command(module, arguments)

    def _training_resume_dataset(self) -> Path | None:
        """Resolve the exact dataset used by a resumable training checkpoint.

        A cloned full-pipeline run must not regenerate source data: doing so can
        change bytes and therefore invalidate the checkpoint's dataset
        fingerprint. External datasets remain explicitly configured, while a
        generated dataset is recovered from the checkpoint's owning run.
        """

        training_requested = self.config.workflow == "train" or self.config.training.enabled
        checkpoint_value = self.config.training.resume_checkpoint
        if not training_requested or not checkpoint_value:
            return None

        if self.config.input.dataset_paths:
            candidate = Path(self.config.input.dataset_paths[0]).expanduser().resolve()
            return candidate.parent if candidate.is_file() else candidate

        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise WorkerError(f"恢复 checkpoint 不存在：{checkpoint}")
        if checkpoint.parent.name != "checkpoints":
            raise WorkerError("恢复生成数据时 checkpoint 必须来自原运行目录的 checkpoints 文件夹")

        original_run = checkpoint.parent.parent.resolve()
        original_config_path = original_run / "run.json"
        try:
            original_config = RunConfig.load(original_config_path)
        except ValueError as exc:
            raise WorkerError(f"无法验证 checkpoint 所属的原运行目录：{exc}") from exc
        if Path(original_config.output.run_dir).expanduser().resolve() != original_run:
            raise WorkerError("checkpoint 所属运行配置指向了另一个目录")

        generated_dataset = original_run / "artifacts" / "dataset"
        if not _completed_generation_is_valid(generated_dataset):
            raise WorkerError("原运行的生成数据不完整或校验失败，不能安全恢复训练")
        return generated_dataset.resolve()

    def _prediction_command(self, checkpoint: Path, sample: Path, index: int) -> list[str]:
        return _module_command(
            "spimaging.testing.predict",
            (
                "--checkpoint", str(checkpoint),
                "--sample_file", str(sample),
                "--output_npz", str(self.layout.gallery / f"sample_{index:05d}.npz"),
                "--output_fig", str(self.layout.gallery / f"sample_{index:05d}.png"),
                "--overwrite",
            ),
        )

    def _evaluation_command(self, checkpoints: tuple[Path, ...], dataset_dir: Path) -> list[str]:
        arguments: list[str] = []
        labels = self.config.evaluation.labels or tuple(path.stem for path in checkpoints)
        for checkpoint, label in zip(checkpoints, labels):
            arguments.extend(("--checkpoint", str(checkpoint), "--label", label))
        arguments.extend(
            (
                "--dataset_dir", str(dataset_dir),
                "--output_dir", str(self.layout.metrics),
                "--figure_index", str(self.config.evaluation.figure_index),
                "--overwrite",
            )
        )
        return _module_command("spimaging.testing.evaluate", arguments)

    def _resolve_dataset(self) -> Path:
        if self.config.workflow == "evaluate" and self.config.evaluation.dataset_dir:
            return Path(self.config.evaluation.dataset_dir).expanduser().resolve()
        if self.config.input.dataset_paths:
            candidate = Path(self.config.input.dataset_paths[0]).expanduser().resolve()
            if candidate.is_file():
                return candidate.parent
            return candidate
        explicit_sample = self.config.prediction.sample_file or self.config.input.sample_file
        if explicit_sample:
            return Path(explicit_sample).expanduser().resolve().parent
        return _public_demo_dataset()

    def _resolve_checkpoint(self) -> Path:
        candidates = [
            self.config.prediction.checkpoint,
            *self.config.evaluation.checkpoints,
            *self.config.input.checkpoint_paths,
        ]
        for value in candidates:
            if value:
                return Path(value).expanduser().resolve()
        for name in ("best.pt", "last.pt", "cancelled.pt"):
            path = self.layout.checkpoints / name
            if path.is_file():
                return path
        public_checkpoint = REPOSITORY_ROOT / "public_demo" / "checkpoint" / "simple3d_synthetic.pt"
        if public_checkpoint.is_file():
            return public_checkpoint.resolve()
        raise WorkerError("没有可用 checkpoint；请先训练或选择 checkpoint。")

    def _selected_samples(self, dataset_dir: Path) -> list[tuple[int, Path]]:
        samples = _list_samples(dataset_dir)
        if not samples:
            raise WorkerError(f"数据集中没有 NPZ 样本：{dataset_dir}")
        requested = self.config.visualization.sample_indices
        if requested:
            indices = requested[:12]
        else:
            indices = tuple(range(min(self.config.visualization.sample_count, len(samples))))
        invalid = [index for index in indices if index >= len(samples)]
        if invalid:
            raise WorkerError(f"可视化样本索引超出范围：{invalid}")
        return [(index, samples[index]) for index in indices]

    def _collect_results(self) -> None:
        assert self.manifest is not None
        excluded = {
            self.layout.config.resolve(),
            self.layout.events.resolve(),
            self.layout.result.resolve(),
            self.layout.log.resolve(),
            self.layout.cancel_request.resolve(),
            self._terminal_reserve.resolve(),
        }
        self.manifest.artifacts.clear()
        for path in sorted(self.layout.root.rglob("*")):
            if not path.is_file() or path.resolve() in excluded:
                continue
            # Matplotlib creates implementation-detail caches next to figures
            # when a CLI is run in an isolated environment.  They are neither
            # experiment results nor portable metadata and must not leak into
            # the result manifest or user exports.
            if ".matplotlib-cache" in path.parts:
                continue
            relative = self.storage.relative_to_root(path)
            suffix = path.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
                kind = "image"
            elif suffix in {".pt", ".pth"}:
                kind = "checkpoint"
            elif suffix in {".csv", ".json"}:
                kind = "metrics" if "metric" in path.name or "summary" in path.name else "metadata"
            elif suffix == ".npz":
                kind = "prediction" if "gallery" in path.parts else "dataset"
            else:
                kind = "file"
            self.manifest.add_artifact(path.stem, relative, kind)
        for candidate in (
            self.layout.metrics / "metrics_summary.json",
            self.layout.artifacts / "demo" / "demo_summary.json",
        ):
            if not candidate.is_file():
                continue
            try:
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = "evaluation" if candidate.name == "metrics_summary.json" else "demo"
            self.manifest.metrics[key] = loaded

    def _run_noop(self) -> None:
        for index in range(1, 4):
            if self.cancel.requested:
                raise WorkerCancelled("任务已由用户取消")
            self.events.emit(
                EventType.STAGE_PROGRESS,
                {"stage": "noop", "current": index, "total": 3, "percent": index * 100 // 3},
            )
            time.sleep(0.02)

    def execute(self) -> int:
        try:
            self.cancel.clear_file()
            self.manifest = self.storage.prepare()
            self._storage_prepared = True
            self._create_terminal_reserve()
            self.history.upsert(self.config, "preparing")
            self.start_control_reader()
            self._set_status("running")
            if self.config.workflow == "noop":
                self._run_noop()
            elif self.config.workflow == "quick_demo":
                dataset = self._resolve_dataset()
                self.run_command(
                    "quick_demo",
                    _module_command(
                        "spimaging.demo",
                        (
                            "--dataset_dir", str(dataset),
                            "--output_dir", str(self.layout.artifacts / "demo"),
                            "--overwrite",
                        ),
                    ),
                )
            else:
                dataset = self._training_resume_dataset()
                generation_requested = (
                    self.config.workflow == "generate" or self.config.generation.enabled
                )
                if dataset is not None and generation_requested:
                    self.events.emit(
                        EventType.WARNING,
                        {
                            "stage": "generate",
                            "message": "恢复训练将复用原运行数据集，已跳过重新生成。",
                        },
                    )
                elif generation_requested:
                    dataset = self.layout.artifacts / "dataset"
                    partial_exists = partial_directory_for(dataset).is_dir()
                    if (
                        self.config.generation.resume
                        and not partial_exists
                        and _completed_generation_is_valid(dataset)
                    ):
                        self.events.emit(
                            EventType.WARNING,
                            {"stage": "generate", "message": "已校验并复用完整的生成数据。"},
                        )
                    else:
                        self.run_command("generate", self._generation_command(dataset))
                if dataset is None:
                    dataset = self._resolve_dataset()

                if self.config.workflow in {"inspect", "full_pipeline"}:
                    self.run_command("inspect", self._verify_command(dataset))

                training_requested = self.config.workflow == "train" or self.config.training.enabled
                if training_requested:
                    self.run_command("train", self._training_command(dataset))

                prediction_requested = self.config.workflow == "predict" or self.config.prediction.enabled
                evaluation_requested = self.config.workflow == "evaluate" or self.config.evaluation.enabled

                checkpoint: Path | None = None
                if prediction_requested or evaluation_requested:
                    checkpoint = self._resolve_checkpoint()

                if prediction_requested and checkpoint is not None:
                    explicit_sample = self.config.prediction.sample_file or self.config.input.sample_file
                    if explicit_sample:
                        selected = [(0, Path(explicit_sample).expanduser().resolve())]
                    else:
                        selected = self._selected_samples(dataset)
                    for ordinal, (sample_index, sample) in enumerate(selected, 1):
                        self.events.emit(
                            EventType.SAMPLE,
                            {"stage": "predict", "index": ordinal, "total": len(selected), "sample": sample.name},
                        )
                        self.run_command(
                            f"predict_{sample_index:05d}",
                            self._prediction_command(checkpoint, sample, sample_index),
                        )

                if evaluation_requested:
                    configured = self.config.evaluation.checkpoints or self.config.input.checkpoint_paths
                    checkpoints = tuple(Path(value).expanduser().resolve() for value in configured)
                    if not checkpoints and checkpoint is not None:
                        checkpoints = (checkpoint,)
                    evaluation_dataset = dataset
                    if self.config.evaluation.dataset_dir:
                        evaluation_dataset = Path(
                            self.config.evaluation.dataset_dir
                        ).expanduser().resolve()
                    self.run_command(
                        "evaluate",
                        self._evaluation_command(checkpoints, evaluation_dataset),
                    )

            try:
                self._collect_results()
            except Exception as exc:
                error = {"category": type(exc).__name__, "message": str(exc)}
                self._finalize_terminal(
                    "failed",
                    error=error,
                    collect_results=False,
                )
                return 1
            self._finalize_terminal("succeeded", collect_results=False)
            return 0
        except WorkerCancelled as exc:
            self._finalize_terminal("cancelled", message=str(exc))
            return 130
        except Exception as exc:
            error = {"category": type(exc).__name__, "message": str(exc)}
            self._finalize_terminal("failed", error=error)
            return 1


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = RunConfig.load(args.config)
    except ValueError as exc:
        build_parser().exit(2, f"spimaging-worker: error: {exc}\n")
    runtime = WorkerRuntime(config, stream=sys.stdout)
    raise SystemExit(runtime.execute())


if __name__ == "__main__":
    main()
