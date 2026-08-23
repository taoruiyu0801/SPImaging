from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtWidgets import QApplication

from spimaging.appcore.config import RunConfig
from spimaging.desktop.controller import WorkerController, normalize_worker_python
from spimaging.desktop.models import (
    ApplicationPaths,
    DesktopSettings,
    PublicDemoAssets,
    RunProgressState,
    SettingsStore,
)
from spimaging.desktop.pages import ExperimentPage, ResultsPage
from spimaging.desktop.widgets import DynamicParameterForm, SampleInspectorDialog
from spimaging.desktop.window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


def _paths(tmp_path: Path) -> ApplicationPaths:
    return ApplicationPaths(
        root=tmp_path,
        runs=tmp_path / "runs",
        cache=tmp_path / "cache",
        history_db=tmp_path / "history.sqlite3",
        settings_file=tmp_path / "settings.json",
    )


def test_main_window_builds_six_pages_offscreen(qapp, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    window = MainWindow(paths=paths, settings_store=SettingsStore(paths.settings_file, paths))
    assert window.stack.count() == 6
    assert set(window.pages) == {"home", "experiment", "run", "results", "history", "settings"}
    window.navigate("results")
    assert window.stack.currentWidget() is window.results_page
    assert window.nav_buttons["results"].isChecked()
    window.close()


def test_experiment_page_has_separate_algorithm_selectors_and_dynamic_forms(qapp, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    settings = DesktopSettings(runs_dir=str(paths.runs), cache_dir=str(paths.cache))
    page = ExperimentPage(paths, settings)
    assert page.simulation_combo.count() == 4
    assert page.reconstruction_combo.count() == 4  # supervised selector
    assert page.simulation_combo.currentData() == "single"
    assert page.reconstruction_combo.currentData() == "simple3d"
    page.training_mode_combo.setCurrentIndex(page.training_mode_combo.findData("self_supervised_spisr"))
    assert page.reconstruction_combo.count() == 1
    assert page.reconstruction_combo.currentData() == "spisr"
    assert "time_scale" in page.training_parameters.controls
    assert "tv_weight" not in page.training_parameters.controls


def test_experiment_page_builds_valid_full_pipeline_config(qapp, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    settings = DesktopSettings(runs_dir=str(paths.runs), cache_dir=str(paths.cache))
    page = ExperimentPage(paths, settings)
    demo = PublicDemoAssets.discover()
    page.dataset_picker.set_path(demo.dataset_dir)
    page.output_picker.set_path(paths.runs)
    page.sample_count_spin.setValue(12)
    config = page.build_config()
    assert config.workflow == "full_pipeline"
    assert config.input.dataset_paths == (str(demo.dataset_dir),)
    assert config.training.enabled
    assert config.visualization.sample_count == 12


def test_parameter_form_advanced_controls_are_collapsible(qapp) -> None:
    from spimaging.desktop.models import ParameterFormState

    form = DynamicParameterForm(ParameterFormState("reconstruction", "simple3d"))
    assert form.advanced_group.isHidden()
    form.advanced_toggle.setChecked(True)
    assert not form.advanced_group.isHidden()
    assert form.values()["epochs"] == 1


def test_nullable_volume_defaults_and_medium_visibility(qapp) -> None:
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

    from spimaging.desktop.models import ParameterFormState

    form = DynamicParameterForm(ParameterFormState("simulation", "volume_scattering"))
    extinction = form.controls["volume_extinction_coeff"]
    assert isinstance(extinction, QDoubleSpinBox)
    assert extinction.specialValueText() == "使用介质默认值"
    assert form.values()["volume_extinction_coeff"] is None
    extinction.setValue(extinction.minimum() + extinction.singleStep())
    assert form.values()["volume_extinction_coeff"] == pytest.approx(0.0)
    extinction.setValue(extinction.minimum())
    assert form.controls["volume_water_front_boost"].isHidden()
    assert not form.controls["volume_fog_front_boost"].isHidden()

    medium = form.controls["volume_medium_type"]
    assert isinstance(medium, QComboBox)
    medium.setCurrentText("water")
    assert not form.controls["volume_water_front_boost"].isHidden()
    assert form.controls["volume_fog_front_boost"].isHidden()


def test_public_sample_inspector_constructs_offscreen(qapp) -> None:
    demo = PublicDemoAssets.discover()
    dialog = SampleInspectorDialog(demo.samples[0])
    assert "样本浏览" in dialog.windowTitle()
    assert dialog.findChildren(DynamicParameterForm) == []
    dialog.close()


def test_results_page_opens_manifest_and_renders_gallery(qapp, tmp_path: Path) -> None:
    from spimaging.appcore.storage import RunStorage
    import numpy as np

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.savez_compressed(
        dataset / "sample_00000.npz",
        counts=np.ones((8, 4, 4), dtype=np.uint16),
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_m=np.ones((4, 4), dtype=np.float32),
    )
    run = tmp_path / "run"
    config = RunConfig.new("full_pipeline", run, input={"dataset_paths": [str(dataset)]})
    storage = RunStorage(config)
    manifest = storage.prepare()
    prediction = run / "gallery" / "sample_00000.npz"
    np.savez_compressed(
        prediction,
        pred_depth_m=np.ones((4, 4), dtype=np.float32),
        target_depth_m=np.ones((4, 4), dtype=np.float32),
        abs_error_m=np.zeros((4, 4), dtype=np.float32),
    )
    manifest.add_artifact("sample_00000", "gallery/sample_00000.npz", "prediction", sample_index=0)
    manifest.set_status("succeeded")
    storage.write_result(manifest)

    page = ResultsPage()
    page.load_result(run)
    qapp.processEvents()
    assert page.model is not None
    assert page.status_badge.text() == "成功"
    assert page.gallery_layout.count() >= 2  # one card plus stretch


def test_worker_python_prefers_windowed_interpreter_on_windows(tmp_path: Path) -> None:
    pythonw = tmp_path / "pythonw.exe"
    python = tmp_path / "python.exe"
    pythonw.touch()
    python.touch()
    assert normalize_worker_python(pythonw) == str(pythonw)
    expected = pythonw if os.name == "nt" else python
    assert normalize_worker_python(python) == str(expected)


def test_controller_reports_malformed_protocol_line(qapp) -> None:
    controller = WorkerController()
    warnings: list[str] = []
    logs: list[str] = []
    controller.protocol_warning.connect(warnings.append)
    controller.log_received.connect(logs.append)
    controller._consume_line("not-json")
    assert warnings
    assert logs == ["not-json"]


def test_controller_executes_noop_worker_via_qprocess(qapp, tmp_path: Path) -> None:
    run_dir = tmp_path / "noop"
    config = RunConfig.new(
        "noop",
        run_dir,
        output={"history_db": str(tmp_path / "history.sqlite3")},
    )
    controller = WorkerController()
    if os.name == "nt" and Path(sys.executable).with_name("pythonw.exe").is_file():
        assert Path(controller.python_executable).name.lower() == "pythonw.exe"
    completed: list[tuple[str, int]] = []
    loop = QEventLoop()

    def finish(status: str, exit_code: int) -> None:
        completed.append((status, exit_code))
        loop.quit()

    controller.completed.connect(finish)
    controller.start(config)

    def timeout() -> None:
        if controller.is_running:
            controller.process.kill()
        loop.quit()

    QTimer.singleShot(15_000, timeout)
    loop.exec()
    assert completed == [("succeeded", 0)]
    assert controller.progress is not None
    assert controller.progress.percent == 100
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "result_manifest.json").is_file()


def test_controller_forced_cancel_persists_manifest_and_history(qapp, tmp_path: Path) -> None:
    from spimaging.appcore.history import HistoryStore
    from spimaging.appcore.storage import RunStorage

    run_dir = tmp_path / "cancelled"
    history_path = tmp_path / "history.sqlite3"
    config = RunConfig.new(
        "noop",
        run_dir,
        output={"history_db": str(history_path)},
    )
    storage = RunStorage(config)
    manifest = storage.prepare()
    manifest.set_status("running")
    storage.write_result(manifest)
    HistoryStore(history_path).upsert(config, "running")
    controller = WorkerController()
    controller._config = config
    controller._progress = RunProgressState(config.run_id, status="running")
    controller._cancel_requested = True

    controller._on_finished(130, QProcess.ExitStatus.CrashExit)

    assert storage.load_result().status == "cancelled"
    assert HistoryStore(history_path).list()[0].status == "cancelled"
