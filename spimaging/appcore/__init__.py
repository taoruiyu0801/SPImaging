"""Shared contracts and runtime services for the SPImaging desktop application."""

from spimaging.appcore.config import (
    ComputeConfig,
    EvaluationConfig,
    GenerationConfig,
    InputConfig,
    OutputConfig,
    PredictionConfig,
    RunConfig,
    TrainingConfig,
    VisualizationConfig,
)
from spimaging.appcore.events import EventType, EventWriter, WorkerEvent
from spimaging.appcore.specs import (
    ALGORITHM_REGISTRY,
    SIMULATION_ALGORITHMS,
    RECONSTRUCTION_ALGORITHMS,
    AlgorithmSpec,
    ParameterSpec,
)
from spimaging.appcore.storage import ResultManifest, RunLayout, RunStorage

__all__ = [
    "ALGORITHM_REGISTRY",
    "SIMULATION_ALGORITHMS",
    "RECONSTRUCTION_ALGORITHMS",
    "AlgorithmSpec",
    "ComputeConfig",
    "EvaluationConfig",
    "EventType",
    "EventWriter",
    "GenerationConfig",
    "InputConfig",
    "OutputConfig",
    "ParameterSpec",
    "PredictionConfig",
    "ResultManifest",
    "RunConfig",
    "RunLayout",
    "RunStorage",
    "TrainingConfig",
    "VisualizationConfig",
    "WorkerEvent",
]
