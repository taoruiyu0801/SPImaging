"""Versioned run configuration for desktop and worker processes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping
import uuid

from spimaging.appcore.specs import (
    GENERATION_COMMON_PARAMETERS,
    get_algorithm,
    validate_training_parameters,
)


RUN_CONFIG_SCHEMA_VERSION = 1
WORKFLOWS = {
    "noop",
    "quick_demo",
    "generate",
    "inspect",
    "train",
    "predict",
    "evaluate",
    "full_pipeline",
}
COMPUTE_PREFERENCES = {"auto", "cuda", "cpu"}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _tuple_of_strings(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label}必须是文本列表")
    return tuple(value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label}必须是对象")
    return dict(value)


def _bool_value(value: Any, label: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{label}必须是布尔值")
    return value


def _section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _mapping(data.get(key, {}), key)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{label}包含未知字段：{', '.join(unknown)}")


@dataclass(frozen=True)
class InputConfig:
    dataset_paths: tuple[str, ...] = ()
    source_path: str | None = None
    sample_file: str | None = None
    checkpoint_paths: tuple[str, ...] = ()
    checkpoint_labels: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputConfig":
        _reject_unknown(
            data,
            {"dataset_paths", "source_path", "sample_file", "checkpoint_paths", "checkpoint_labels"},
            "input",
        )
        source_path = data.get("source_path")
        sample_file = data.get("sample_file")
        if source_path is not None and not isinstance(source_path, str):
            raise ValueError("input.source_path 必须是文本或 null")
        if sample_file is not None and not isinstance(sample_file, str):
            raise ValueError("input.sample_file 必须是文本或 null")
        result = cls(
            dataset_paths=_tuple_of_strings(data.get("dataset_paths"), "input.dataset_paths"),
            source_path=source_path,
            sample_file=sample_file,
            checkpoint_paths=_tuple_of_strings(data.get("checkpoint_paths"), "input.checkpoint_paths"),
            checkpoint_labels=_tuple_of_strings(data.get("checkpoint_labels"), "input.checkpoint_labels"),
        )
        if result.checkpoint_labels and len(result.checkpoint_labels) != len(result.checkpoint_paths):
            raise ValueError("checkpoint_labels 数量必须与 checkpoint_paths 相同")
        return result


@dataclass(frozen=True)
class GenerationConfig:
    enabled: bool = False
    dataset_mode: str = "raw"
    surface_model: str = "single"
    parameters: dict[str, Any] = field(default_factory=dict)
    resume: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationConfig":
        _reject_unknown(data, {"enabled", "dataset_mode", "surface_model", "parameters", "resume"}, "generation")
        result = cls(
            enabled=_bool_value(data.get("enabled"), "generation.enabled", False),
            dataset_mode=str(data.get("dataset_mode", "raw")),
            surface_model=str(data.get("surface_model", "single")),
            parameters=_mapping(data.get("parameters"), "generation.parameters"),
            resume=_bool_value(data.get("resume"), "generation.resume", True),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.dataset_mode not in {"labeled", "raw", "middlebury"}:
            raise ValueError(f"未知数据源模式：{self.dataset_mode}")
        algorithm = get_algorithm("simulation", self.surface_model)
        common = {parameter.name: parameter for parameter in GENERATION_COMMON_PARAMETERS}
        specific = {parameter.name: parameter for parameter in algorithm.parameters}
        unknown = sorted(set(self.parameters) - set(common) - set(specific))
        if unknown:
            raise ValueError(f"生成配置包含未知参数：{', '.join(unknown)}")
        for name, value in self.parameters.items():
            (common.get(name) or specific[name]).validate(value)


@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = False
    model: str = "simple3d"
    preset: str = "quick"
    parameters: dict[str, Any] = field(default_factory=dict)
    resume_checkpoint: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        _reject_unknown(data, {"enabled", "model", "preset", "parameters", "resume_checkpoint"}, "training")
        checkpoint = data.get("resume_checkpoint")
        if checkpoint is not None and not isinstance(checkpoint, str):
            raise ValueError("training.resume_checkpoint 必须是文本或 null")
        result = cls(
            enabled=_bool_value(data.get("enabled"), "training.enabled", False),
            model=str(data.get("model", "simple3d")),
            preset=str(data.get("preset", "quick")),
            parameters=_mapping(data.get("parameters"), "training.parameters"),
            resume_checkpoint=checkpoint,
        )
        validate_training_parameters(result.model, result.preset, result.parameters)
        return result

    def resolved_parameters(self) -> dict[str, Any]:
        return validate_training_parameters(self.model, self.preset, self.parameters)


@dataclass(frozen=True)
class PredictionConfig:
    enabled: bool = False
    checkpoint: str | None = None
    sample_file: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredictionConfig":
        _reject_unknown(data, {"enabled", "checkpoint", "sample_file"}, "prediction")
        checkpoint = data.get("checkpoint")
        sample_file = data.get("sample_file")
        if checkpoint is not None and not isinstance(checkpoint, str):
            raise ValueError("prediction.checkpoint 必须是文本或 null")
        if sample_file is not None and not isinstance(sample_file, str):
            raise ValueError("prediction.sample_file 必须是文本或 null")
        return cls(_bool_value(data.get("enabled"), "prediction.enabled", False), checkpoint, sample_file)


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = False
    checkpoints: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    dataset_dir: str | None = None
    figure_index: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationConfig":
        _reject_unknown(data, {"enabled", "checkpoints", "labels", "dataset_dir", "figure_index"}, "evaluation")
        dataset_dir = data.get("dataset_dir")
        if dataset_dir is not None and not isinstance(dataset_dir, str):
            raise ValueError("evaluation.dataset_dir 必须是文本或 null")
        result = cls(
            enabled=_bool_value(data.get("enabled"), "evaluation.enabled", False),
            checkpoints=_tuple_of_strings(data.get("checkpoints"), "evaluation.checkpoints"),
            labels=_tuple_of_strings(data.get("labels"), "evaluation.labels"),
            dataset_dir=dataset_dir,
            figure_index=int(data.get("figure_index", 0)),
        )
        if result.figure_index < 0:
            raise ValueError("evaluation.figure_index 不能为负数")
        if result.labels and len(result.labels) != len(result.checkpoints):
            raise ValueError("evaluation.labels 数量必须与 checkpoints 相同")
        return result


@dataclass(frozen=True)
class VisualizationConfig:
    sample_count: int = 4
    sample_indices: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualizationConfig":
        _reject_unknown(data, {"sample_count", "sample_indices"}, "visualization")
        raw_indices = data.get("sample_indices", ())
        if not isinstance(raw_indices, (list, tuple)):
            raise ValueError("visualization.sample_indices 必须是整数列表")
        indices = tuple(int(index) for index in raw_indices)
        result = cls(int(data.get("sample_count", 4)), indices)
        if not 1 <= result.sample_count <= 12:
            raise ValueError("可视化样本数必须在 1 到 12 之间")
        if any(index < 0 for index in result.sample_indices):
            raise ValueError("可视化样本索引不能为负数")
        if len(result.sample_indices) > 12:
            raise ValueError("最多选择 12 个可视化样本")
        return result


@dataclass(frozen=True)
class ComputeConfig:
    preference: str = "auto"
    gpu_index: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComputeConfig":
        _reject_unknown(data, {"preference", "gpu_index"}, "compute")
        result = cls(str(data.get("preference", "auto")), int(data.get("gpu_index", 0)))
        if result.preference not in COMPUTE_PREFERENCES:
            raise ValueError(f"未知设备偏好：{result.preference}")
        if result.gpu_index < 0:
            raise ValueError("GPU 索引不能为负数")
        return result


@dataclass(frozen=True)
class OutputConfig:
    run_dir: str
    history_db: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputConfig":
        _reject_unknown(data, {"run_dir", "history_db"}, "output")
        run_dir = data.get("run_dir")
        history_db = data.get("history_db")
        if not isinstance(run_dir, str) or not run_dir.strip():
            raise ValueError("output.run_dir 不能为空")
        if history_db is not None and not isinstance(history_db, str):
            raise ValueError("output.history_db 必须是文本或 null")
        return cls(run_dir, history_db)


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    run_id: str
    workflow: str
    created_at: str
    input: InputConfig
    generation: GenerationConfig
    training: TrainingConfig
    prediction: PredictionConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    compute: ComputeConfig
    output: OutputConfig
    display_name: str = "SPImaging 实验"

    @classmethod
    def new(
        cls,
        workflow: str,
        run_dir: str | Path,
        *,
        display_name: str = "SPImaging 实验",
        **sections: Any,
    ) -> "RunConfig":
        data = {
            "schema_version": RUN_CONFIG_SCHEMA_VERSION,
            "run_id": str(uuid.uuid4()),
            "workflow": workflow,
            "created_at": _now(),
            "display_name": display_name,
            "input": sections.get("input", {}),
            "generation": sections.get("generation", {}),
            "training": sections.get("training", {}),
            "prediction": sections.get("prediction", {}),
            "evaluation": sections.get("evaluation", {}),
            "visualization": sections.get("visualization", {}),
            "compute": sections.get("compute", {}),
            "output": {"run_dir": str(run_dir), **sections.get("output", {})},
        }
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        _reject_unknown(
            data,
            {
                "schema_version", "run_id", "workflow", "created_at", "display_name",
                "input", "generation", "training", "prediction", "evaluation",
                "visualization", "compute", "output",
            },
            "run config",
        )
        schema_version = int(data.get("schema_version", 0))
        if schema_version != RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"不支持的 RunConfig schema_version：{schema_version}")
        run_id = str(data.get("run_id", ""))
        try:
            uuid.UUID(run_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("run_id 必须是有效 UUID") from exc
        workflow = str(data.get("workflow", ""))
        if workflow not in WORKFLOWS:
            raise ValueError(f"未知工作流：{workflow}")
        created_at = str(data.get("created_at", ""))
        if not created_at:
            raise ValueError("created_at 不能为空")
        display_name = str(data.get("display_name", "SPImaging 实验")).strip()
        if not display_name:
            raise ValueError("display_name 不能为空")
        result = cls(
            schema_version=schema_version,
            run_id=run_id,
            workflow=workflow,
            created_at=created_at,
            input=InputConfig.from_dict(_section(data, "input")),
            generation=GenerationConfig.from_dict(_section(data, "generation")),
            training=TrainingConfig.from_dict(_section(data, "training")),
            prediction=PredictionConfig.from_dict(_section(data, "prediction")),
            evaluation=EvaluationConfig.from_dict(_section(data, "evaluation")),
            visualization=VisualizationConfig.from_dict(_section(data, "visualization")),
            compute=ComputeConfig.from_dict(_section(data, "compute")),
            output=OutputConfig.from_dict(_section(data, "output")),
            display_name=display_name,
        )
        result.validate_execution_requirements()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取运行配置 {path}：{exc}") from exc
        if not isinstance(data, Mapping):
            raise ValueError("运行配置根节点必须是对象")
        return cls.from_dict(data)

    def validate_execution_requirements(self) -> None:
        has_dataset = bool(self.input.dataset_paths)
        generated_dataset = self.generation.enabled
        if self.workflow == "generate" and not self.input.source_path:
            raise ValueError("数据生成工作流需要 input.source_path")
        if self.workflow in {"inspect", "train"} and not has_dataset:
            raise ValueError(f"{self.workflow} 工作流需要 input.dataset_paths")
        if self.workflow == "predict" and not (
            self.prediction.checkpoint or self.input.checkpoint_paths
        ):
            raise ValueError("预测工作流需要 checkpoint")
        if self.workflow == "predict" and not (
            self.prediction.sample_file or self.input.sample_file
        ):
            raise ValueError("预测工作流需要 sample_file")
        if self.workflow == "evaluate" and not (
            self.evaluation.checkpoints or self.input.checkpoint_paths
        ):
            raise ValueError("评估工作流需要 checkpoint")
        if self.workflow == "evaluate" and not (
            self.evaluation.dataset_dir or has_dataset
        ):
            raise ValueError("评估工作流需要数据集")
        if self.workflow == "full_pipeline" and not (has_dataset or generated_dataset):
            raise ValueError("完整工作流需要已有数据集或启用数据生成")
        if self.workflow == "full_pipeline" and generated_dataset and not self.input.source_path:
            raise ValueError("启用数据生成时需要 input.source_path")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
