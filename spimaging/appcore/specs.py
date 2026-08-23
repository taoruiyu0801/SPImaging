"""Algorithm and parameter specifications shared by CLI adapters and desktop forms."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


class ParameterType(str, Enum):
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"
    CHOICE = "choice"


@dataclass(frozen=True)
class ParameterSpec:
    """One validated, optionally conditional GUI/worker parameter."""

    name: str
    label: str
    kind: ParameterType
    default: Any
    help: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    advanced: bool = False
    visible_when: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    cli_flag: str | None = None
    false_cli_flag: str | None = None

    def is_visible(self, values: Mapping[str, Any]) -> bool:
        return all(values.get(key) in allowed for key, allowed in self.visible_when.items())

    def validate(self, value: Any) -> Any:
        if self.kind is ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"{self.label}必须是布尔值")
            return value
        if self.kind is ParameterType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{self.label}必须是整数")
            numeric = float(value)
        elif self.kind is ParameterType.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{self.label}必须是数值")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{self.label}必须是有限数值")
            value = numeric
        elif self.kind in (ParameterType.STRING, ParameterType.CHOICE):
            if not isinstance(value, str):
                raise ValueError(f"{self.label}必须是文本")
            if self.kind is ParameterType.CHOICE and value not in self.choices:
                raise ValueError(f"{self.label}必须是以下选项之一：{', '.join(self.choices)}")
            return value
        else:  # pragma: no cover - exhaustive guard
            raise ValueError(f"未知参数类型：{self.kind}")

        if self.minimum is not None and numeric < self.minimum:
            raise ValueError(f"{self.label}不能小于 {self.minimum:g}")
        if self.maximum is not None and numeric > self.maximum:
            raise ValueError(f"{self.label}不能大于 {self.maximum:g}")
        return value

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        result["visible_when"] = {
            key: list(values) for key, values in self.visible_when.items()
        }
        result["choices"] = list(self.choices)
        return result


@dataclass(frozen=True)
class AlgorithmSpec:
    """A simulation or reconstruction algorithm exposed by the product."""

    key: str
    label: str
    category: str
    description: str
    parameters: tuple[ParameterSpec, ...] = ()
    method_family: str | None = None
    bundled_checkpoint: bool = False

    def parameter_defaults(self) -> dict[str, Any]:
        return {parameter.name: parameter.default for parameter in self.parameters}

    def visible_parameters(self, values: Mapping[str, Any]) -> tuple[ParameterSpec, ...]:
        merged = {**self.parameter_defaults(), **dict(values)}
        return tuple(parameter for parameter in self.parameters if parameter.is_visible(merged))

    def validate_parameters(self, values: Mapping[str, Any]) -> dict[str, Any]:
        known = {parameter.name: parameter for parameter in self.parameters}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"{self.label}包含未知参数：{', '.join(unknown)}")
        merged = self.parameter_defaults()
        merged.update(values)
        return {
            parameter.name: parameter.validate(merged[parameter.name])
            for parameter in self.parameters
            if parameter.is_visible(merged)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "method_family": self.method_family,
            "bundled_checkpoint": self.bundled_checkpoint,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


def _integer(name: str, label: str, default: int, minimum: int = 0, **kwargs) -> ParameterSpec:
    return ParameterSpec(
        name,
        label,
        ParameterType.INTEGER,
        default,
        minimum=minimum,
        cli_flag=f"--{name}",
        **kwargs,
    )


def _number(name: str, label: str, default: float, minimum: float = 0.0, **kwargs) -> ParameterSpec:
    return ParameterSpec(
        name,
        label,
        ParameterType.NUMBER,
        default,
        minimum=minimum,
        cli_flag=f"--{name}",
        **kwargs,
    )


def _choice(name: str, label: str, default: str, choices: tuple[str, ...], **kwargs) -> ParameterSpec:
    return ParameterSpec(
        name,
        label,
        ParameterType.CHOICE,
        default,
        choices=choices,
        cli_flag=f"--{name}",
        **kwargs,
    )


def _boolean(name: str, label: str, default: bool, flag: str | None = None, **kwargs) -> ParameterSpec:
    return ParameterSpec(
        name,
        label,
        ParameterType.BOOLEAN,
        default,
        cli_flag=flag or f"--{name}",
        **kwargs,
    )


GENERATION_COMMON_PARAMETERS = (
    _integer("param_idx", "光子参数组", 10, 1, maximum=10),
    _integer("res", "输出分辨率", 64, 1),
    _integer("bins", "时间 bin 数", 1024, 1),
    _number("bin_size", "bin 宽度（秒）", 80e-12, minimum=1e-15, advanced=True),
    _integer("limit", "最多生成样本", 4, 1),
    _boolean("save_x", "保存调试代理 x", False, advanced=True),
    _boolean("save_clean_transient", "保存无噪瞬态", False, advanced=True),
)

SIMULATION_ALGORITHMS: dict[str, AlgorithmSpec] = {
    "single": AlgorithmSpec(
        "single",
        "单表面（Single）",
        "simulation",
        "使用 DeepInverse SinglePhotonLidar 生成单回波测量。",
        (_number("irf_sigma", "IRF 标准差", 2.0, minimum=1e-9),),
    ),
    "neighborhood_mix": AlgorithmSpec(
        "neighborhood_mix",
        "邻域混合（Neighborhood Mix）",
        "simulation",
        "通过空间邻域混合产生多返回瞬态。",
        (
            _integer("mix_kernel_size", "混合核大小", 5, 1),
            _number("mix_sigma_xy", "空间高斯 σ", 1.0, minimum=1e-9),
            _number("mix_time_sigma_bins", "时间高斯 σ", 2.0, minimum=1e-9),
        ),
    ),
    "translucent_layer": AlgorithmSpec(
        "translucent_layer",
        "半透明层（Translucent Layer）",
        "simulation",
        "模拟半透明前层与后景的双返回。",
        (
            _choice(
                "translucent_front_type",
                "前层形状",
                "flat",
                ("flat", "sloped", "sinusoidal"),
            ),
            _number("translucent_front_depth", "前层深度（米）", 1.0, minimum=1e-9),
            _number(
                "translucent_front_depth_x_slope",
                "X 方向深度斜率",
                0.0,
                minimum=-1000.0,
                maximum=1000.0,
                advanced=True,
            ),
            _number(
                "translucent_front_depth_y_slope",
                "Y 方向深度斜率",
                0.0,
                minimum=-1000.0,
                maximum=1000.0,
                advanced=True,
            ),
            _number("translucent_front_depth_amplitude", "正弦深度振幅", 0.1),
            _number("translucent_front_signal_ratio", "前层反射比", 0.25),
            _number("translucent_transmission", "后景透射率", 0.6, maximum=1.0),
            _number("translucent_time_sigma_bins", "时间高斯 σ", 2.0, minimum=1e-9),
        ),
    ),
    "volume_scattering": AlgorithmSpec(
        "volume_scattering",
        "体散射（Volume Scattering）",
        "simulation",
        "模拟雾或水体中的路径散射和表面返回。",
        (
            _choice("volume_medium_type", "介质", "fog", ("fog", "water")),
            _number("volume_extinction_coeff", "消光系数", 0.15, advanced=True),
            _number("volume_backscatter_ratio", "后向散射强度", 0.2, advanced=True),
            _number("volume_scatter_depth_fraction", "散射深度比例", 0.9, minimum=1e-9, maximum=1.0),
            _integer("volume_num_steps", "路径积分步数", 64, 1),
            _number("volume_time_sigma_bins", "时间高斯 σ", 2.0, minimum=1e-9),
            _number("volume_range_weight_power", "距离权重指数", 1.0, advanced=True),
            _number("volume_water_front_boost", "水体近场增强", 1.5, advanced=True),
            _number("volume_fog_front_boost", "雾近场增强", 1.0, advanced=True),
        ),
    ),
}

TRAINING_COMMON_PARAMETERS = (
    _integer("epochs", "训练轮数", 20, 1),
    _integer("batch_size", "批大小", 1, 1),
    _number("lr", "学习率", 1e-3, minimum=1e-12),
    _number("weight_decay", "权重衰减", 1e-5, advanced=True),
    _number("val_fraction", "验证集比例", 0.2, maximum=0.999999),
    _integer("seed", "随机种子", 0, 0, maximum=4294967295, advanced=True),
    _integer("num_workers", "数据加载进程", 0, 0, advanced=True),
    _integer("max_samples", "最多使用样本", 0, 0, advanced=True),
)

RECONSTRUCTION_ALGORITHMS: dict[str, AlgorithmSpec] = {
    "simple3d": AlgorithmSpec(
        "simple3d",
        "Simple3D",
        "reconstruction",
        "轻量 3D 卷积基线；公开版内置预训练模型。",
        (
            _integer("base_channels", "基础通道数", 8, 1),
            _integer("temporal_downsample", "时间降采样", 1, 1),
            _number("target_sigma_bins", "目标高斯宽度", 2.0, minimum=1e-9, advanced=True),
            _choice("target_source", "监督目标", "depth", ("depth", "clean"), advanced=True),
            _boolean("no_log_counts", "禁用 log1p 计数压缩", False, advanced=True),
            _number("tv_weight", "TV 损失权重", 0.0, advanced=True),
            _integer("early_stopping_patience", "早停耐心轮数", 0, 0, advanced=True),
            _number("early_stopping_min_delta", "早停最小改进（米）", 1e-4, advanced=True),
        ),
        method_family="supervised",
        bundled_checkpoint=True,
    ),
    "prsnet": AlgorithmSpec(
        "prsnet",
        "PRSNet",
        "reconstruction",
        "残差 3D 光子重建网络。",
        (
            _integer("num_blocks", "残差块数量", 10, 1),
            _integer("temporal_downsample", "时间降采样", 1, 1),
            _number("target_sigma_bins", "目标高斯宽度", 2.0, minimum=1e-9, advanced=True),
            _choice("target_source", "监督目标", "depth", ("depth", "clean"), advanced=True),
            _boolean("no_log_counts", "禁用 log1p 计数压缩", False, advanced=True),
            _number("tv_weight", "TV 损失权重", 0.0, advanced=True),
        ),
        method_family="supervised",
    ),
    "penonlocal": AlgorithmSpec(
        "penonlocal",
        "PENonLocal",
        "reconstruction",
        "包含 non-local 模块的 3D 光子重建网络。",
        (
            _integer("num_blocks", "非局部块数量", 10, 1),
            _integer("temporal_downsample", "时间降采样", 1, 1),
            _number("target_sigma_bins", "目标高斯宽度", 2.0, minimum=1e-9, advanced=True),
            _choice("target_source", "监督目标", "depth", ("depth", "clean"), advanced=True),
            _boolean("no_log_counts", "禁用 log1p 计数压缩", False, advanced=True),
            _number("tv_weight", "TV 损失权重", 0.0, advanced=True),
        ),
        method_family="supervised",
    ),
    "stin": AlgorithmSpec(
        "stin",
        "STIN",
        "reconstruction",
        "固定结构的时空交互网络。",
        (
            _integer("temporal_downsample", "时间降采样", 1, 1),
            _number("target_sigma_bins", "目标高斯宽度", 2.0, minimum=1e-9, advanced=True),
            _choice("target_source", "监督目标", "depth", ("depth", "clean"), advanced=True),
            _boolean("no_log_counts", "禁用 log1p 计数压缩", False, advanced=True),
            _number("tv_weight", "TV 损失权重", 0.0, advanced=True),
        ),
        method_family="supervised",
    ),
    "spisr": AlgorithmSpec(
        "spisr",
        "SPISR（自监督）",
        "reconstruction",
        "使用 PUKL 和等变损失的自监督时空超分网络。",
        (
            _integer("base_channels", "基础通道数", 16, 1),
            _integer("num_blocks", "处理块数量", 4, 1),
            _integer("time_scale", "时间超分倍率", 2, 1),
            _integer("spatial_scale", "空间超分倍率", 2, 1),
            _integer("temporal_downsample", "时间降采样", 8, 1),
            _integer("spatial_downsample", "空间降采样", 2, 1),
            _number("gamma", "PUKL 噪声参数 γ", 0.005, minimum=1e-12, advanced=True),
            _number("tau", "有限差分尺度 τ", 1e-3, minimum=1e-12, advanced=True),
            _number("alpha", "等变损失权重 α", 1.0, advanced=True),
            _number("poisson_scale", "泊松扰动尺度", 1.0, minimum=1e-12, advanced=True),
            _integer("max_shift", "最大时间平移", 8, 0, advanced=True),
            _boolean("no_normalize", "禁用 LR 归一化", False, advanced=True),
        ),
        method_family="self_supervised_spisr",
    ),
}

ALGORITHM_REGISTRY = {**SIMULATION_ALGORITHMS, **RECONSTRUCTION_ALGORITHMS}

TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "epochs": 1,
        "batch_size": 1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "val_fraction": 0.5,
        "seed": 0,
        "num_workers": 0,
        "max_samples": 2,
    },
    "standard": {parameter.name: parameter.default for parameter in TRAINING_COMMON_PARAMETERS},
    "custom": {parameter.name: parameter.default for parameter in TRAINING_COMMON_PARAMETERS},
}


def get_algorithm(category: str, key: str) -> AlgorithmSpec:
    registry = SIMULATION_ALGORITHMS if category == "simulation" else RECONSTRUCTION_ALGORITHMS
    try:
        return registry[key]
    except KeyError as exc:
        raise ValueError(f"未知{category}算法：{key}") from exc


def validate_training_parameters(
    model: str,
    preset: str,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    if preset not in TRAINING_PRESETS:
        raise ValueError(f"未知训练预设：{preset}")
    algorithm = get_algorithm("reconstruction", model)
    common_specs = {parameter.name: parameter for parameter in TRAINING_COMMON_PARAMETERS}
    algorithm_specs = {parameter.name: parameter for parameter in algorithm.parameters}
    unknown = sorted(set(values) - set(common_specs) - set(algorithm_specs))
    if unknown:
        raise ValueError(f"训练配置包含未知参数：{', '.join(unknown)}")
    merged = dict(TRAINING_PRESETS[preset])
    merged.update(algorithm.parameter_defaults())
    merged.update(values)
    validated = {
        name: spec.validate(merged[name]) for name, spec in common_specs.items()
    }
    validated.update(algorithm.validate_parameters({
        name: merged[name] for name in algorithm_specs if name in merged
    }))
    return validated

