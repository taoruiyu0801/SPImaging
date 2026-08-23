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
        if not (key in {"max_samples", "early_stopping_patience"} and value == 0)
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
        if self.config.input.dataset_paths:
            candidate = Path(self.config.input.dataset_paths[0]).expanduser().resolve()
            if candidate.is_file():
                return candidate.parent
            return candidate
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
        }
        self.manifest.artifacts.clear()
        for path in sorted(self.layout.root.rglob("*")):
            if not path.is_file() or path.resolve() in excluded:
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
        self.cancel.clear_file()
        self.manifest = self.storage.prepare()
        self.history.upsert(self.config, "preparing")
        self.start_control_reader()
        try:
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
                dataset: Path | None = None
                if self.config.workflow == "generate" or self.config.generation.enabled:
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
                if self.config.workflow == "full_pipeline":
                    prediction_requested = True
                    evaluation_requested = True

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
                    self.run_command("evaluate", self._evaluation_command(checkpoints, dataset))

            self._collect_results()
            self._set_status("succeeded")
            self.events.emit(
                EventType.COMPLETED,
                {"status": "succeeded", "result_manifest": str(self.layout.result)},
            )
            return 0
        except WorkerCancelled as exc:
            self._collect_results()
            self._set_status("cancelled", message=str(exc))
            self.events.emit(EventType.COMPLETED, {"status": "cancelled"})
            return 130
        except Exception as exc:
            error = {"category": type(exc).__name__, "message": str(exc)}
            self._collect_results()
            self._set_status("failed", error=error)
            self.events.emit(EventType.ERROR, error)
            self.events.emit(EventType.COMPLETED, {"status": "failed"})
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
