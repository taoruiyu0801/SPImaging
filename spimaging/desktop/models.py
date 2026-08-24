"""Desktop-facing models built on the versioned :mod:`spimaging.appcore` contracts.

This module deliberately has no Qt dependency.  Keeping configuration, history,
progress and gallery discovery here makes the desktop testable in headless
environments and gives future front ends the same behaviour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping
import uuid

from spimaging.appcore.config import RunConfig
from spimaging.appcore.events import WorkerEvent
from spimaging.appcore.history import HistoryRecord, HistoryStore
from spimaging.appcore.specs import (
    GENERATION_COMMON_PARAMETERS,
    RECONSTRUCTION_ALGORITHMS,
    SIMULATION_ALGORITHMS,
    TRAINING_COMMON_PARAMETERS,
    TRAINING_PRESETS,
    AlgorithmSpec,
    ParameterSpec,
    get_algorithm,
    training_preset_values,
    validate_training_parameters,
)
from spimaging.appcore.storage import ArtifactRecord, ResultManifest, safe_relative_path
from spimaging.generation.recovery import partial_directory_for


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORKFLOW_LABELS = {
    "quick_demo": "快速体验",
    "full_pipeline": "完整流程",
    "generate": "生成数据",
    "inspect": "检查样本",
    "train": "训练模型",
    "predict": "单样本预测",
    "evaluate": "批量评估 / 模型比较",
    "noop": "连接自检",
}

STATUS_LABELS = {
    "preparing": "准备",
    "running": "运行",
    "cancelling": "取消中",
    "succeeded": "成功",
    "failed": "失败",
    "cancelled": "已取消",
    "interrupted": "意外中断",
}

DEVICE_LABELS = {
    "auto": "自动（优先 NVIDIA）",
    "cuda": "NVIDIA GPU",
    "cpu": "CPU",
}


def cuda_required() -> bool:
    """Return whether this desktop was launched by the CUDA-only bootstrap."""

    return os.environ.get("SPIMAGING_CUDA_REQUIRED", "").strip() == "1"


def desktop_device_labels() -> dict[str, str]:
    return {"cuda": DEVICE_LABELS["cuda"]} if cuda_required() else dict(DEVICE_LABELS)

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "interrupted"}
RESUMABLE_STATUSES = {"failed", "cancelled", "interrupted"}
_SAMPLE_INDEX = re.compile(r"(?:sample|input|prediction)[_-]?(\d+)", re.IGNORECASE)
_SUPERVISED_METRIC = re.compile(r"(?:^|[_\s])(mae|rmse|absrel|abs_rel|error)(?:$|[_\s(])", re.I)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _inside(root: Path, relative: str | Path) -> Path:
    """Resolve a portable relative path and reject path escape."""

    safe = safe_relative_path(relative)
    candidate = (root / Path(*safe.split("/"))).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:  # defensive; safe_relative_path already rejects ``..``
        raise ValueError(f"路径位于允许目录之外：{relative}") from exc
    return candidate


@dataclass(frozen=True)
class ApplicationPaths:
    """Per-user storage locations used by the source desktop and packaged app."""

    root: Path
    runs: Path
    cache: Path
    history_db: Path
    settings_file: Path

    @classmethod
    def default(cls) -> "ApplicationPaths":
        local = os.environ.get("LOCALAPPDATA")
        root = (Path(local) / "SPImaging") if local else (Path.home() / ".spimaging")
        root = root.expanduser().resolve()
        return cls(
            root=root,
            runs=root / "runs",
            cache=root / "cache",
            history_db=root / "history.sqlite3",
            settings_file=root / "settings.json",
        )

    def ensure(self) -> None:
        for directory in (self.root, self.runs, self.cache):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PublicDemoAssets:
    """Verified paths advertised by the synthetic public asset manifest."""

    root: Path
    manifest_path: Path
    dataset_dir: Path
    checkpoint: Path
    samples: tuple[Path, ...]
    release: str
    available: bool
    reason: str = ""

    @classmethod
    def discover(cls, project_root: str | Path | None = None) -> "PublicDemoAssets":
        root = Path(project_root or PROJECT_ROOT).expanduser().resolve() / "public_demo"
        manifest_path = root / "manifest.json"
        unavailable = cls(root, manifest_path, root / "dataset", root / "checkpoint", (), "", False)
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping) or int(raw.get("schema_version", 0)) != 1:
                raise ValueError("公开演示清单版本不受支持")
            dataset = raw.get("dataset")
            checkpoint = raw.get("checkpoint")
            if not isinstance(dataset, Mapping) or not isinstance(checkpoint, Mapping):
                raise ValueError("公开演示清单缺少数据集或 checkpoint")
            dataset_dir = _inside(root, str(dataset.get("path", "")))
            checkpoint_path = _inside(root, str(checkpoint.get("path", "")))
            sample_items = dataset.get("samples", [])
            if not isinstance(sample_items, list):
                raise ValueError("公开演示样本清单格式错误")
            samples = tuple(
                _inside(root, str(item.get("path", "")))
                for item in sample_items
                if isinstance(item, Mapping)
            )
            expected = int(dataset.get("sample_count", len(samples)))
            if expected != len(samples) or not samples:
                raise ValueError("公开演示样本数量与清单不一致")
            missing = [path for path in (*samples, checkpoint_path) if not path.is_file()]
            if missing or not dataset_dir.is_dir():
                raise ValueError(f"公开演示资产不完整：{missing[0] if missing else dataset_dir}")
            return cls(
                root=root,
                manifest_path=manifest_path,
                dataset_dir=dataset_dir,
                checkpoint=checkpoint_path,
                samples=samples,
                release=str(raw.get("release", "")),
                available=True,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return replace(unavailable, reason=str(exc))


class ParameterFormState:
    """State and validation for one registry-driven parameter form."""

    def __init__(self, category: str, algorithm: str, *, preset: str = "quick") -> None:
        if category not in {"simulation", "reconstruction"}:
            raise ValueError(f"未知参数表单类别：{category}")
        self.category = category
        self.algorithm = algorithm
        self.preset = preset
        self.values: dict[str, Any] = {}
        self.reset()

    @property
    def algorithm_spec(self) -> AlgorithmSpec:
        return get_algorithm(self.category, self.algorithm)

    @property
    def common_specs(self) -> tuple[ParameterSpec, ...]:
        return GENERATION_COMMON_PARAMETERS if self.category == "simulation" else TRAINING_COMMON_PARAMETERS

    @property
    def specs(self) -> tuple[ParameterSpec, ...]:
        return (*self.common_specs, *self.algorithm_spec.parameters)

    def reset(self) -> None:
        if self.category == "reconstruction":
            self.values = self.algorithm_spec.parameter_defaults()
            self.values.update(training_preset_values(self.algorithm, self.preset))
        else:
            self.values = {item.name: item.default for item in self.common_specs}
            self.values.update(self.algorithm_spec.parameter_defaults())

    def set_algorithm(self, algorithm: str) -> None:
        previous_algorithm = self.algorithm
        previous_common = {
            item.name: self.values[item.name]
            for item in self.common_specs
            if item.name in self.values
        }
        get_algorithm(self.category, algorithm)
        self.algorithm = algorithm
        self.reset()
        if self.category == "reconstruction":
            old_defaults = training_preset_values(previous_algorithm, self.preset)
            new_defaults = training_preset_values(algorithm, self.preset)
            self.values.update(
                {
                    name: value
                    for name, value in previous_common.items()
                    if value != old_defaults.get(name) or old_defaults.get(name) == new_defaults.get(name)
                }
            )
        else:
            self.values.update(previous_common)

    def apply_preset(self, preset: str) -> None:
        if self.category != "reconstruction":
            raise ValueError("仿真参数不支持训练预设")
        if preset not in TRAINING_PRESETS:
            raise ValueError(f"未知训练预设：{preset}")
        if preset == "custom":
            self.preset = preset
            return
        self.preset = preset
        self.reset()

    def set_value(self, name: str, value: Any) -> None:
        known = {item.name: item for item in self.specs}
        if name not in known:
            raise ValueError(f"未知参数：{name}")
        self.values[name] = known[name].validate(value)

    def visible_specs(self, *, include_advanced: bool = False) -> tuple[ParameterSpec, ...]:
        values = dict(self.values)
        return tuple(
            item
            for item in self.specs
            if (include_advanced or not item.advanced) and item.is_visible(values)
        )

    def resolved_values(self) -> dict[str, Any]:
        if self.category == "reconstruction":
            return validate_training_parameters(self.algorithm, self.preset, self.values)
        common = {item.name: item for item in self.common_specs}
        algorithm = self.algorithm_spec
        result = {
            name: spec.validate(self.values.get(name, spec.default))
            for name, spec in common.items()
        }
        specific_values = {
            item.name: self.values.get(item.name, item.default)
            for item in algorithm.parameters
        }
        result.update(algorithm.validate_parameters(specific_values))
        return result


@dataclass
class ExperimentRequest:
    """Mutable UI draft that produces exactly one validated ``RunConfig``."""

    workflow: str = "full_pipeline"
    display_name: str = "SPImaging 实验"
    dataset_paths: tuple[str, ...] = ()
    source_path: str | None = None
    sample_file: str | None = None
    checkpoint_paths: tuple[str, ...] = ()
    checkpoint_labels: tuple[str, ...] = ()
    generation_enabled: bool = False
    dataset_mode: str = "raw"
    simulation_model: str = "single"
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    training_enabled: bool = True
    reconstruction_model: str = "simple3d"
    training_preset: str = "quick"
    training_parameters: dict[str, Any] = field(default_factory=dict)
    resume_checkpoint: str | None = None
    prediction_enabled: bool = True
    prediction_checkpoint: str | None = None
    evaluation_enabled: bool = True
    evaluation_checkpoints: tuple[str, ...] = ()
    evaluation_labels: tuple[str, ...] = ()
    figure_index: int = 0
    sample_count: int = 4
    sample_indices: tuple[int, ...] = ()
    device: str = "auto"
    gpu_index: int = 0

    def to_run_config(
        self,
        run_dir: str | Path,
        *,
        history_db: str | Path | None = None,
    ) -> RunConfig:
        workflow = self.workflow
        generation_enabled = self.generation_enabled or workflow == "generate"
        training_enabled = self.training_enabled or workflow == "train"
        prediction_enabled = self.prediction_enabled or workflow == "predict"
        evaluation_enabled = self.evaluation_enabled or workflow == "evaluate"
        if workflow in {"generate", "inspect", "train", "predict", "evaluate", "quick_demo", "noop"}:
            # A single-purpose workflow must not accidentally execute unrelated stages.
            generation_enabled = workflow == "generate"
            training_enabled = workflow == "train"
            prediction_enabled = workflow == "predict"
            evaluation_enabled = workflow == "evaluate"

        prediction_checkpoint = self.prediction_checkpoint
        if prediction_checkpoint is None and self.checkpoint_paths:
            prediction_checkpoint = self.checkpoint_paths[0]
        evaluation_checkpoints = self.evaluation_checkpoints or self.checkpoint_paths
        evaluation_labels = self.evaluation_labels or self.checkpoint_labels
        evaluation_dataset = self.dataset_paths[0] if self.dataset_paths else None

        generation_form = ParameterFormState("simulation", self.simulation_model)
        generation_form.values.update(self.generation_parameters)
        training_form = ParameterFormState(
            "reconstruction", self.reconstruction_model, preset=self.training_preset
        )
        training_form.values.update(self.training_parameters)

        return RunConfig.new(
            workflow,
            run_dir,
            display_name=self.display_name.strip(),
            input={
                "dataset_paths": list(self.dataset_paths),
                "source_path": self.source_path,
                "sample_file": self.sample_file,
                "checkpoint_paths": list(self.checkpoint_paths),
                "checkpoint_labels": list(self.checkpoint_labels),
            },
            generation={
                "enabled": generation_enabled,
                "dataset_mode": self.dataset_mode,
                "surface_model": self.simulation_model,
                "parameters": generation_form.resolved_values(),
                "resume": True,
            },
            training={
                "enabled": training_enabled,
                "model": self.reconstruction_model,
                "preset": self.training_preset,
                "parameters": training_form.resolved_values(),
                "resume_checkpoint": self.resume_checkpoint,
            },
            prediction={
                "enabled": prediction_enabled,
                "checkpoint": prediction_checkpoint,
                "sample_file": self.sample_file,
            },
            evaluation={
                "enabled": evaluation_enabled,
                "checkpoints": list(evaluation_checkpoints),
                "labels": list(evaluation_labels),
                "dataset_dir": evaluation_dataset,
                "figure_index": self.figure_index,
            },
            visualization={
                "sample_count": self.sample_count,
                "sample_indices": list(self.sample_indices),
            },
            compute={"preference": self.device, "gpu_index": self.gpu_index},
            output={"history_db": None if history_db is None else str(history_db)},
        )


def next_run_directory(root: str | Path, display_name: str = "run") -> Path:
    """Return a collision-resistant, readable run directory without creating it."""

    safe_name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", display_name).strip("-_")
    safe_name = safe_name[:32] or "run"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(root).expanduser().resolve() / f"{stamp}-{safe_name}-{uuid.uuid4().hex[:8]}"


def clone_run_config(
    config: RunConfig,
    run_dir: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    target_epochs: int | None = None,
) -> RunConfig:
    """Clone a historical config with a new identity and optional safe resume."""

    data = config.to_dict()
    data["run_id"] = str(uuid.uuid4())
    data["created_at"] = _now()
    data["output"]["run_dir"] = str(Path(run_dir).expanduser().resolve())
    if resume_checkpoint is None:
        data["training"]["resume_checkpoint"] = None
    else:
        if not config.training.enabled and config.workflow != "train":
            raise ValueError("该历史任务不包含训练阶段，不能恢复训练")
        original_epochs = int(config.training.resolved_parameters()["epochs"])
        new_epochs = original_epochs + 1 if target_epochs is None else int(target_epochs)
        if new_epochs <= original_epochs:
            raise ValueError(f"恢复训练的目标轮数必须大于原目标 {original_epochs}")
        data["training"]["parameters"]["epochs"] = new_epochs
        data["training"]["resume_checkpoint"] = str(Path(resume_checkpoint).expanduser().resolve())
        if config.generation.enabled:
            generated_dataset = (
                Path(config.output.run_dir).expanduser().resolve() / "artifacts" / "dataset"
            )
            if not generated_dataset.is_dir() or not any(generated_dataset.glob("*.npz")):
                raise ValueError("原任务没有可复用的完整生成数据，不能安全恢复训练")
            # A resumed checkpoint is fingerprinted against the original
            # generated samples.  Re-generating into the new run would make
            # recovery dependent on simulator/runtime details and may fail the
            # content fingerprint check.
            data["generation"]["enabled"] = False
            data["generation"]["resume"] = False
            data["input"]["dataset_paths"] = [str(generated_dataset)]
            data["evaluation"]["dataset_dir"] = str(generated_dataset)
    return RunConfig.from_dict(data)


@dataclass
class MetricPoint:
    step: int
    value: float


@dataclass
class RunProgressState:
    """Deterministic projection of the worker event stream for the Run page."""

    run_id: str
    status: str = "preparing"
    stage: str = ""
    current: int = 0
    total: int = 0
    percent: int = 0
    last_seq: int = 0
    message: str = ""
    metrics: dict[str, list[MetricPoint]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def apply(self, event: WorkerEvent) -> None:
        if event.run_id != self.run_id:
            raise ValueError("事件 run_id 与当前任务不一致")
        if event.seq <= self.last_seq:
            raise ValueError("事件序号必须严格递增")
        self.last_seq = event.seq
        payload = event.payload

        if event.type == "state":
            status = str(payload.get("status", self.status))
            if status in STATUS_LABELS:
                self.status = status
        elif event.type == "stage_started":
            self.stage = str(payload.get("stage", ""))
            self.current = self.total = self.percent = 0
        elif event.type in {"stage_progress", "epoch", "batch", "sample"}:
            self.stage = str(payload.get("stage", self.stage))
            self.current = _first_int(payload, "current", "index", "batch", "epoch", default=self.current)
            self.total = _first_int(payload, "total", "batches", "epochs", default=self.total)
            if "percent" in payload:
                self.percent = max(0, min(100, int(payload["percent"])))
            elif self.total > 0:
                self.percent = max(0, min(100, round(self.current * 100 / self.total)))
            self._capture_metrics(payload)
        elif event.type == "metric":
            self._capture_metrics(payload)
        elif event.type == "warning":
            message = str(payload.get("message", payload))
            self.warnings.append(message)
            self.message = message
        elif event.type == "error":
            self.error = str(payload.get("message", payload))
            self.message = self.error
        elif event.type == "log":
            self.message = str(payload.get("message", ""))
        elif event.type == "completed":
            status = str(payload.get("status", self.status))
            if status in STATUS_LABELS:
                self.status = status
            self.percent = 100 if self.status == "succeeded" else self.percent

    def _capture_metrics(self, payload: Mapping[str, Any]) -> None:
        step = _first_int(payload, "global_step", "step", "batch", "epoch", default=self.last_seq)
        containers: list[Mapping[str, Any]] = [payload]
        nested = payload.get("metrics")
        if isinstance(nested, Mapping):
            containers.append(nested)
        for container in containers:
            for key, value in container.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                lower = str(key).lower()
                if any(token in lower for token in ("loss", "mae", "rmse", "absrel")):
                    self.metrics.setdefault(str(key), []).append(MetricPoint(step, float(value)))


def _first_int(payload: Mapping[str, Any], *names: str, default: int) -> int:
    for name in names:
        value = payload.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return default


@dataclass(frozen=True)
class GallerySample:
    index: int
    title: str
    input_overview: Any | None = None
    target_depth: Any | None = None
    prediction_depth: Any | None = None
    absolute_error: Any | None = None
    histogram: Any | None = None
    layers: Mapping[str, Any] = field(default_factory=dict)
    composite_image: Path | None = None
    labeled: bool = False
    message: str = ""


@dataclass(frozen=True)
class SampleInspection:
    """Safe, presentation-ready projection of one SPAD sample archive."""

    path: Path
    rgb: Any | None
    count_map: Any | None
    depth: Any | None
    histogram: Any | None
    layers: Mapping[str, Any]
    fields: tuple[str, ...]


def load_sample_inspection(path: str | Path) -> SampleInspection:
    from spimaging.training_common.security import inspect_npz_archive, load_npz_arrays

    sample_path = Path(path).expanduser().resolve()
    members = inspect_npz_archive(sample_path, required_keys=("counts",))
    wanted = [key for key in ("rgb", "counts", "depth_m") if key in members]
    layer_names = [
        key
        for key, member in members.items()
        if len(member.shape) == 2
        and any(token in key for token in ("front", "back", "volume", "scatter", "transmission"))
    ][:8]
    loaded = load_npz_arrays(sample_path, keys=(*wanted, *layer_names))
    import numpy as np

    counts = np.asarray(loaded["counts"])
    if counts.ndim == 3:
        count_map = counts.sum(axis=0)
        y, x = counts.shape[-2] // 2, counts.shape[-1] // 2
        histogram = counts[:, y, x]
    else:
        count_map = counts
        histogram = None
    return SampleInspection(
        path=sample_path,
        rgb=loaded.get("rgb"),
        count_map=count_map,
        depth=loaded.get("depth_m"),
        histogram=histogram,
        layers={key: loaded[key] for key in layer_names},
        fields=tuple(sorted(members)),
    )


class ResultGalleryModel:
    """Validated result-manifest reader and lazy 1--12 sample gallery source."""

    def __init__(self, run_dir: Path, config: RunConfig, manifest: ResultManifest) -> None:
        self.run_dir = run_dir.resolve()
        self.config = config
        self.manifest = manifest
        self._artifacts = tuple(manifest.artifacts)

    @classmethod
    def load(cls, value: str | Path) -> "ResultGalleryModel":
        path = Path(value).expanduser().resolve()
        run_dir = path if path.is_dir() else path.parent
        manifest_path = run_dir / "result_manifest.json"
        config_path = run_dir / "run.json"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取结果清单：{exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("结果清单根节点必须是对象")
        manifest = ResultManifest.from_dict(raw)
        config = RunConfig.load(config_path)
        if manifest.run_id != config.run_id:
            raise ValueError("结果清单与运行配置不属于同一任务")
        return cls(run_dir, config, manifest)

    def artifact_path(self, artifact: ArtifactRecord) -> Path:
        candidate = _inside(self.run_dir, artifact.path)
        return candidate

    @property
    def artifacts(self) -> tuple[tuple[ArtifactRecord, Path], ...]:
        return tuple((item, self.artifact_path(item)) for item in self._artifacts)

    @property
    def sample_indices(self) -> tuple[int, ...]:
        indices: set[int] = set()
        for artifact in self._artifacts:
            if artifact.sample_index is not None:
                indices.add(artifact.sample_index)
                continue
            match = _SAMPLE_INDEX.search(Path(artifact.path).stem)
            if match:
                indices.add(int(match.group(1)))
        if not indices:
            dataset = self._dataset_samples()
            indices.update(range(min(len(dataset), self.config.visualization.sample_count)))
        if not indices and any(item.kind == "image" for item in self._artifacts):
            indices.add(0)
        return tuple(sorted(indices))

    def selected_indices(self, count: int = 4) -> tuple[int, ...]:
        if not 1 <= int(count) <= 12:
            raise ValueError("画廊样本数必须在 1 到 12 之间")
        return self.sample_indices[: int(count)]

    def _dataset_samples(self) -> tuple[Path, ...]:
        if self.config.input.sample_file:
            sample = Path(self.config.input.sample_file).expanduser().resolve()
            return (sample,) if sample.is_file() else ()
        if self.config.prediction.sample_file:
            sample = Path(self.config.prediction.sample_file).expanduser().resolve()
            return (sample,) if sample.is_file() else ()
        for raw in self.config.input.dataset_paths:
            path = Path(raw).expanduser().resolve()
            if path.is_file() and path.suffix.lower() == ".npz":
                return (path,)
            if path.is_dir():
                samples = tuple(sorted(path.glob("sample_*.npz")) or sorted(path.glob("*.npz")))
                if samples:
                    return samples
        if self.config.workflow == "quick_demo":
            demo = PublicDemoAssets.discover()
            return demo.samples if demo.available else ()
        generated = self.run_dir / "artifacts" / "dataset"
        if generated.is_dir():
            return tuple(sorted(generated.glob("sample_*.npz")) or sorted(generated.glob("*.npz")))
        return ()

    def _prediction_for(self, index: int) -> Path | None:
        candidates: list[Path] = []
        for artifact, path in self.artifacts:
            if path.suffix.lower() != ".npz":
                continue
            match = _SAMPLE_INDEX.search(path.stem)
            if artifact.sample_index == index or (match and int(match.group(1)) == index):
                candidates.append(path)
            elif index == 0 and "prediction" in path.stem.lower():
                candidates.append(path)
        if not candidates:
            direct = self.run_dir / "gallery" / f"sample_{index:05d}.npz"
            if direct.is_file():
                candidates.append(direct)
        return next((item for item in candidates if item.is_file()), None)

    def _composite_for(self, index: int) -> Path | None:
        matches: list[Path] = []
        for artifact, path in self.artifacts:
            if artifact.kind != "image" or not path.is_file():
                continue
            found = _SAMPLE_INDEX.search(path.stem)
            if artifact.sample_index == index or (found and int(found.group(1)) == index):
                matches.append(path)
            elif index == 0 and any(token in path.stem.lower() for token in ("comparison", "prediction")):
                matches.append(path)
        return matches[0] if matches else None

    def load_sample(self, index: int) -> GallerySample:
        if index < 0:
            raise ValueError("样本索引不能为负数")
        source_samples = self._dataset_samples()
        source = source_samples[index] if index < len(source_samples) else None
        prediction_path = self._prediction_for(index)
        composite = self._composite_for(index)
        input_overview = target = prediction = error = histogram = None
        layers: dict[str, Any] = {}
        messages: list[str] = []

        try:
            if source is not None and source.is_file():
                input_overview, target, histogram, layers = _load_source_visuals(source)
        except (OSError, ValueError) as exc:
            messages.append(f"输入样本无法预览：{exc}")
        try:
            if prediction_path is not None:
                loaded = _load_prediction_visuals(prediction_path)
                prediction = loaded.get("pred_depth_m")
                target = loaded.get("target_depth_m", target)
                error = loaded.get("abs_error_m")
                if error is None and prediction is not None and target is not None:
                    import numpy as np

                    if tuple(np.asarray(prediction).shape) == tuple(np.asarray(target).shape):
                        error = np.abs(np.asarray(prediction) - np.asarray(target))
        except (OSError, ValueError) as exc:
            messages.append(f"预测产物无法预览：{exc}")
        labeled = target is not None
        if not labeled:
            error = None
        return GallerySample(
            index=index,
            title=f"样本 {index + 1}",
            input_overview=input_overview,
            target_depth=target,
            prediction_depth=prediction,
            absolute_error=error,
            histogram=histogram,
            layers=layers,
            composite_image=composite,
            labeled=labeled,
            message="；".join(messages),
        )

    def filtered_metrics(self, *, labeled: bool) -> dict[str, Any]:
        if labeled:
            return dict(self.manifest.metrics)
        return _drop_supervised_metrics(self.manifest.metrics)

    def compatible_resume_checkpoint(self) -> Path | None:
        if self.manifest.status not in RESUMABLE_STATUSES:
            return None
        for name in ("cancelled.pt", "last.pt", "best.pt"):
            candidate = self.run_dir / "checkpoints" / name
            if candidate.is_file():
                return candidate
        for artifact, path in self.artifacts:
            if artifact.kind == "checkpoint" and path.is_file():
                return path
        return None

    def generation_resume_directory(self) -> Path | None:
        """Return the owned partial generation directory for same-run resume."""

        if self.manifest.status not in RESUMABLE_STATUSES:
            return None
        if self.config.workflow != "generate" and not self.config.generation.enabled:
            return None
        dataset = self.run_dir / "artifacts" / "dataset"
        partial = partial_directory_for(dataset)
        return partial if partial.is_dir() else None

    def resume_available(self) -> bool:
        return (
            self.generation_resume_directory() is not None
            or self.compatible_resume_checkpoint() is not None
        )


def _load_source_visuals(path: Path) -> tuple[Any | None, Any | None, Any | None, dict[str, Any]]:
    inspected = load_sample_inspection(path)
    overview = inspected.rgb if inspected.rgb is not None else inspected.count_map
    return overview, inspected.depth, inspected.histogram, dict(inspected.layers)


def _load_prediction_visuals(path: Path) -> dict[str, Any]:
    from spimaging.training_common.security import inspect_npz_archive, load_npz_arrays

    members = inspect_npz_archive(path)
    keys = [key for key in ("pred_depth_m", "target_depth_m", "abs_error_m") if key in members]
    return load_npz_arrays(path, keys=keys)


def _drop_supervised_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if _SUPERVISED_METRIC.search(str(key)):
            continue
        if isinstance(item, Mapping):
            nested = _drop_supervised_metrics(item)
            if nested:
                result[str(key)] = nested
        else:
            result[str(key)] = item
    return result


class RunHistoryModel:
    """Rebuildable history facade with status recovery on desktop startup."""

    def __init__(self, database: str | Path) -> None:
        self.store = HistoryStore(database)

    def recover_interrupted(self) -> int:
        return self.store.mark_interrupted()

    def list(self, limit: int = 100) -> list[HistoryRecord]:
        return self.store.list(limit=limit)

    def rebuild(self, roots: Iterable[str | Path]) -> tuple[int, list[str]]:
        return self.store.rebuild(roots)


@dataclass
class DesktopSettings:
    schema_version: int = 1
    device: str = "auto"
    gpu_index: int = 0
    runs_dir: str = ""
    cache_dir: str = ""
    update_checks: bool = True
    locale: str = "zh_CN"

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"不支持的设置版本：{self.schema_version}")
        if self.device not in DEVICE_LABELS:
            raise ValueError(f"未知设备偏好：{self.device}")
        if isinstance(self.gpu_index, bool) or int(self.gpu_index) < 0:
            raise ValueError("GPU 索引不能为负数")
        if not self.runs_dir.strip() or not self.cache_dir.strip():
            raise ValueError("运行目录和缓存目录不能为空")
        if not isinstance(self.update_checks, bool):
            raise ValueError("更新检查设置必须是布尔值")


class SettingsStore:
    def __init__(self, path: str | Path, defaults: ApplicationPaths | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.defaults = defaults or ApplicationPaths.default()

    def default_settings(self) -> DesktopSettings:
        return DesktopSettings(
            device="cuda" if cuda_required() else "auto",
            runs_dir=str(self.defaults.runs),
            cache_dir=str(self.defaults.cache),
        )

    def load(self) -> DesktopSettings:
        if not self.path.is_file():
            return self.default_settings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("设置根节点必须是对象")
            allowed = {item.name for item in DesktopSettings.__dataclass_fields__.values()}
            unknown = set(raw) - allowed
            if unknown:
                raise ValueError(f"设置包含未知字段：{', '.join(sorted(unknown))}")
            settings = DesktopSettings(**dict(raw))
            settings.validate()
            if cuda_required() and settings.device != "cuda":
                settings = replace(settings, device="cuda")
            return settings
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # A damaged preferences file must not prevent the workbench opening.
            return self.default_settings()

    def save(self, settings: DesktopSettings) -> None:
        settings.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(asdict(settings), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


__all__ = [
    "ApplicationPaths",
    "DesktopSettings",
    "DEVICE_LABELS",
    "ExperimentRequest",
    "GallerySample",
    "MetricPoint",
    "ParameterFormState",
    "PublicDemoAssets",
    "RESUMABLE_STATUSES",
    "ResultGalleryModel",
    "RunHistoryModel",
    "RunProgressState",
    "SampleInspection",
    "STATUS_LABELS",
    "SettingsStore",
    "TERMINAL_STATUSES",
    "WORKFLOW_LABELS",
    "clone_run_config",
    "cuda_required",
    "desktop_device_labels",
    "load_sample_inspection",
    "next_run_directory",
]
