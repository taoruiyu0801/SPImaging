from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spimaging.appcore.config import RunConfig
from spimaging.appcore.events import EventType, WorkerEvent
from spimaging.appcore.storage import RunStorage
from spimaging.desktop.models import (
    ApplicationPaths,
    DesktopSettings,
    ExperimentRequest,
    ParameterFormState,
    PublicDemoAssets,
    ResultGalleryModel,
    RunProgressState,
    SettingsStore,
    clone_run_config,
    load_sample_inspection,
    next_run_directory,
)


def _sample(path: Path, *, labeled: bool = True) -> None:
    counts = np.arange(8 * 5 * 6, dtype=np.uint16).reshape(8, 5, 6)
    fields = {
        "counts": counts,
        "rgb": np.full((5, 6, 3), 127, dtype=np.uint8),
        "bin_size": np.asarray(8e-11, dtype=np.float64),
    }
    if labeled:
        fields["depth_m"] = np.linspace(0.5, 2.0, 30, dtype=np.float32).reshape(5, 6)
    np.savez_compressed(path, **fields)


def _result_run(tmp_path: Path, *, labeled: bool) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _sample(dataset / "sample_00000.npz", labeled=labeled)
    run_dir = tmp_path / "run"
    config = RunConfig.new(
        "full_pipeline",
        run_dir,
        input={"dataset_paths": [str(dataset)]},
        training={"enabled": False},
        prediction={"enabled": True},
        visualization={"sample_count": 4},
    )
    storage = RunStorage(config)
    manifest = storage.prepare()
    prediction = run_dir / "gallery" / "sample_00000.npz"
    pred = np.full((5, 6), 1.0, dtype=np.float32)
    values = {"pred_depth_m": pred}
    if labeled:
        target = np.full((5, 6), 1.25, dtype=np.float32)
        values.update(target_depth_m=target, abs_error_m=np.abs(pred - target))
    np.savez_compressed(prediction, **values)
    manifest.add_artifact("sample_00000", "gallery/sample_00000.npz", "prediction", sample_index=0)
    manifest.metrics = {"MAE(m)": 0.25, "runtime_seconds": 1.2}
    manifest.set_status("succeeded")
    storage.write_result(manifest)
    return run_dir


def test_public_demo_manifest_is_discovered() -> None:
    demo = PublicDemoAssets.discover()
    assert demo.available, demo.reason
    assert len(demo.samples) == 4
    assert demo.dataset_dir.is_dir()
    assert demo.checkpoint.name == "simple3d_synthetic.pt"


def test_parameter_states_follow_registry_and_presets() -> None:
    simulation = ParameterFormState("simulation", "translucent_layer")
    names = {item.name for item in simulation.visible_specs(include_advanced=True)}
    assert "translucent_front_depth" in names
    assert "base_channels" not in names

    training = ParameterFormState("reconstruction", "simple3d", preset="quick")
    assert training.resolved_values()["epochs"] == 1
    assert training.resolved_values()["max_samples"] == 2
    training.set_algorithm("spisr")
    resolved = training.resolved_values()
    assert resolved["time_scale"] == 2
    assert "tv_weight" not in resolved
    training.apply_preset("standard")
    assert training.resolved_values()["epochs"] == 20


def test_experiment_request_round_trips_through_run_config(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    request = ExperimentRequest(
        workflow="full_pipeline",
        display_name="round trip",
        dataset_paths=(str(dataset),),
        simulation_model="volume_scattering",
        reconstruction_model="spisr",
        training_preset="quick",
        training_parameters={"base_channels": 4, "num_blocks": 1},
        sample_count=12,
        device="cpu",
    )
    config = request.to_run_config(tmp_path / "run", history_db=tmp_path / "history.db")
    loaded = RunConfig.from_dict(json.loads(config.to_json()))
    assert loaded.generation.surface_model == "volume_scattering"
    assert loaded.training.model == "spisr"
    assert loaded.training.resolved_parameters()["base_channels"] == 4
    assert loaded.visualization.sample_count == 12
    assert loaded.compute.preference == "cpu"


def test_sample_inspection_is_pickle_free_and_presentation_ready() -> None:
    demo = PublicDemoAssets.discover()
    inspected = load_sample_inspection(demo.samples[0])
    assert inspected.rgb.shape == (64, 64, 3)
    assert inspected.count_map.shape == (64, 64)
    assert inspected.depth.shape == (64, 64)
    assert inspected.histogram.shape == (1024,)
    assert "counts" in inspected.fields


@pytest.mark.parametrize("labeled", [True, False])
def test_result_gallery_handles_labeled_and_unlabeled_runs(tmp_path: Path, labeled: bool) -> None:
    model = ResultGalleryModel.load(_result_run(tmp_path, labeled=labeled))
    assert model.selected_indices(12) == (0,)
    sample = model.load_sample(0)
    assert sample.labeled is labeled
    assert sample.input_overview is not None
    assert sample.prediction_depth is not None
    if labeled:
        assert sample.target_depth is not None
        assert sample.absolute_error is not None
        assert "MAE(m)" in model.filtered_metrics(labeled=True)
    else:
        assert sample.target_depth is None
        assert sample.absolute_error is None
        assert "MAE(m)" not in model.filtered_metrics(labeled=False)
        assert model.filtered_metrics(labeled=False)["runtime_seconds"] == 1.2


def test_gallery_count_range_is_enforced(tmp_path: Path) -> None:
    model = ResultGalleryModel.load(_result_run(tmp_path, labeled=True))
    with pytest.raises(ValueError, match="1 到 12"):
        model.selected_indices(0)
    with pytest.raises(ValueError, match="1 到 12"):
        model.selected_indices(13)


def test_progress_projection_rejects_cross_run_and_duplicate_events() -> None:
    run_id = "b8dd3596-89f0-45ef-9d08-a5528726ad50"
    state = RunProgressState(run_id)
    state.apply(WorkerEvent.create(run_id, 1, EventType.STATE, {"status": "running"}))
    state.apply(
        WorkerEvent.create(
            run_id,
            2,
            EventType.BATCH,
            {"stage": "train", "batch": 2, "batches": 4, "loss": 0.75},
        )
    )
    assert state.status == "running"
    assert state.percent == 50
    assert state.metrics["loss"][0].value == pytest.approx(0.75)
    with pytest.raises(ValueError, match="严格递增"):
        state.apply(WorkerEvent.create(run_id, 2, EventType.LOG, {}))
    with pytest.raises(ValueError, match="run_id"):
        state.apply(WorkerEvent.create("another", 3, EventType.LOG, {}))


def test_resume_clone_only_increases_target_epochs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    original = RunConfig.new(
        "train",
        tmp_path / "first",
        input={"dataset_paths": [str(dataset)]},
        training={"enabled": True, "model": "simple3d", "preset": "quick"},
    )
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"placeholder")
    cloned = clone_run_config(original, tmp_path / "second", resume_checkpoint=checkpoint)
    assert cloned.run_id != original.run_id
    assert cloned.training.resume_checkpoint == str(checkpoint.resolve())
    assert cloned.training.resolved_parameters()["epochs"] == 2
    with pytest.raises(ValueError, match="必须大于"):
        clone_run_config(original, tmp_path / "bad", resume_checkpoint=checkpoint, target_epochs=1)


def test_settings_round_trip_and_corruption_fallback(tmp_path: Path) -> None:
    paths = ApplicationPaths(
        tmp_path,
        tmp_path / "runs",
        tmp_path / "cache",
        tmp_path / "history.db",
        tmp_path / "settings.json",
    )
    store = SettingsStore(paths.settings_file, paths)
    settings = DesktopSettings(
        device="cpu",
        runs_dir=str(paths.runs),
        cache_dir=str(paths.cache),
        update_checks=False,
    )
    store.save(settings)
    assert store.load() == settings
    paths.settings_file.write_text("{broken", encoding="utf-8")
    fallback = store.load()
    assert fallback.device == "auto"
    assert fallback.runs_dir == str(paths.runs)


def test_next_run_directory_is_readable_and_unique(tmp_path: Path) -> None:
    first = next_run_directory(tmp_path, "我的 实验")
    second = next_run_directory(tmp_path, "我的 实验")
    assert first != second
    assert first.parent == tmp_path.resolve()
    assert "我的-实验" in first.name
