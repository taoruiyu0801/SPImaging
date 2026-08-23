"""Chinese-first pages for the SPImaging experiment workbench."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from spimaging.appcore.config import RunConfig
from spimaging.appcore.diagnostics import export_diagnostic_bundle
from spimaging.appcore.history import HistoryRecord
from spimaging.appcore.specs import RECONSTRUCTION_ALGORITHMS, SIMULATION_ALGORITHMS
from spimaging.appcore.storage import atomic_write_text
from spimaging.desktop.controller import WorkerController
from spimaging.desktop.dependency import require_pyside6
from spimaging.desktop.i18n import tr
from spimaging.desktop.models import (
    DEVICE_LABELS,
    RESUMABLE_STATUSES,
    STATUS_LABELS,
    TERMINAL_STATUSES,
    WORKFLOW_LABELS,
    ApplicationPaths,
    DesktopSettings,
    ExperimentRequest,
    ParameterFormState,
    PublicDemoAssets,
    ResultGalleryModel,
    RunHistoryModel,
    RunProgressState,
    SettingsStore,
    next_run_directory,
)
from spimaging.desktop.widgets import (
    DynamicParameterForm,
    GalleryCard,
    LineChartWidget,
    PageHeader,
    PathPicker,
    SampleInspectorDialog,
    SectionCard,
    StatusBadge,
)


require_pyside6()

from PySide6.QtCore import Qt, QThread, Signal  # noqa: E402
from PySide6.QtGui import QDesktopServices  # noqa: E402
from PySide6.QtCore import QUrl  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _scroll_page(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)
    return scroll


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        child = item.layout()
        if child is not None:
            _clear_layout(child)


def _open_local_path(path: str | Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).expanduser().resolve())))


class HomePage(QWidget):
    quick_demo_requested = Signal()
    new_experiment_requested = Signal()
    open_result_requested = Signal()
    history_requested = Signal()
    open_run_requested = Signal(str)

    def __init__(self, demo: PublicDemoAssets, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 24, 26, 26)
        root.setSpacing(16)
        root.addWidget(
            PageHeader(
                tr("HomePage", "欢迎使用 SPImaging"),
                tr("HomePage", "从合成样例快速验证流程，或创建可复现的单光子成像实验。"),
            )
        )

        actions = QGridLayout()
        actions.setSpacing(12)
        quick = SectionCard(
            tr("HomePage", "快速体验"),
            tr("HomePage", "使用 4 个 CC0 合成样本完成检查、最小训练、预测和评估。"),
        )
        self.demo_status = QLabel()
        self.demo_status.setWordWrap(True)
        self.demo_status.setObjectName("muted")
        if demo.available:
            self.demo_status.setText(
                tr("HomePage", "公开演示资产已就绪") + f" · {len(demo.samples)} samples · {demo.release}"
            )
        else:
            self.demo_status.setText(tr("HomePage", "公开演示资产不可用：") + demo.reason)
        self.quick_button = QPushButton(tr("HomePage", "开始快速体验"))
        self.quick_button.setObjectName("primaryButton")
        self.quick_button.setEnabled(demo.available)
        self.quick_button.clicked.connect(self.quick_demo_requested)
        quick.body_layout.addWidget(self.demo_status)
        quick.body_layout.addWidget(self.quick_button, 0, Qt.AlignmentFlag.AlignLeft)

        create = SectionCard(
            tr("HomePage", "新建实验"),
            tr("HomePage", "选择数据、仿真模型、重建算法、训练方式和参数。"),
        )
        create_button = QPushButton(tr("HomePage", "打开实验工作台"))
        create_button.clicked.connect(self.new_experiment_requested)
        create.body_layout.addWidget(create_button, 0, Qt.AlignmentFlag.AlignLeft)

        open_card = SectionCard(
            tr("HomePage", "打开结果"),
            tr("HomePage", "读取独立运行目录中的清单、指标、预测和画廊。"),
        )
        open_button = QPushButton(tr("HomePage", "选择运行目录…"))
        open_button.clicked.connect(self.open_result_requested)
        open_card.body_layout.addWidget(open_button, 0, Qt.AlignmentFlag.AlignLeft)

        history_card = SectionCard(
            tr("HomePage", "实验历史"),
            tr("HomePage", "恢复中断任务，或复用过去的配置开始新实验。"),
        )
        history_button = QPushButton(tr("HomePage", "查看全部历史"))
        history_button.clicked.connect(self.history_requested)
        history_card.body_layout.addWidget(history_button, 0, Qt.AlignmentFlag.AlignLeft)
        actions.addWidget(quick, 0, 0)
        actions.addWidget(create, 0, 1)
        actions.addWidget(open_card, 1, 0)
        actions.addWidget(history_card, 1, 1)
        root.addLayout(actions)

        recent = SectionCard(tr("HomePage", "最近运行"), tr("HomePage", "双击可直接打开结果。"))
        self.recent_table = QTableWidget(0, 4)
        self.recent_table.setHorizontalHeaderLabels(
            [tr("HomePage", "名称"), tr("HomePage", "工作流"), tr("HomePage", "状态"), tr("HomePage", "更新时间")]
        )
        self.recent_table.horizontalHeader().setStretchLastSection(True)
        self.recent_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.recent_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.recent_table.doubleClicked.connect(self._open_selected)
        recent.body_layout.addWidget(self.recent_table)
        root.addWidget(recent, 1)
        self._records: list[HistoryRecord] = []

    def set_history(self, records: Iterable[HistoryRecord]) -> None:
        self._records = list(records)[:8]
        self.recent_table.setRowCount(len(self._records))
        for row, record in enumerate(self._records):
            values = (
                record.display_name,
                tr("WorkflowLabels", WORKFLOW_LABELS.get(record.workflow, record.workflow)),
                tr("StatusLabels", STATUS_LABELS.get(record.status, record.status)),
                record.updated_at,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, record.run_dir)
                self.recent_table.setItem(row, column, item)

    def _open_selected(self) -> None:
        row = self.recent_table.currentRow()
        if 0 <= row < len(self._records):
            self.open_run_requested.emit(self._records[row].run_dir)


class ExperimentPage(QWidget):
    run_requested = Signal(object)
    config_saved = Signal(str)

    def __init__(
        self,
        paths: ApplicationPaths,
        settings: DesktopSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.paths = paths
        self.settings = settings
        self._last_config: RunConfig | None = None

        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(26, 24, 26, 26)
        root.setSpacing(14)
        root.addWidget(
            PageHeader(
                tr("ExperimentPage", "实验工作台"),
                tr("ExperimentPage", "仿真与重建算法相互独立；参数表只展示当前实现真正使用的选项。"),
            )
        )

        identity = SectionCard(tr("ExperimentPage", "任务与输入"))
        form = QFormLayout()
        self.name_edit = QLineEdit(tr("ExperimentPage", "SPImaging 实验"))
        self.name_edit.setObjectName("displayNameEdit")
        self.workflow_combo = QComboBox()
        self.workflow_combo.setObjectName("workflowCombo")
        for key in ("full_pipeline", "generate", "inspect", "train", "predict", "evaluate"):
            self.workflow_combo.addItem(tr("WorkflowLabels", WORKFLOW_LABELS[key]), key)
        self.dataset_picker = PathPicker(mode="directory", placeholder=tr("ExperimentPage", "已有 NPZ 数据集目录"))
        self.sample_picker = PathPicker(
            mode="open_file", file_filter="NPZ (*.npz)", placeholder=tr("ExperimentPage", "单样本预测文件")
        )
        self.checkpoint_picker = PathPicker(
            mode="open_file", file_filter="PyTorch checkpoint (*.pt *.pth)", placeholder=tr("ExperimentPage", "预测或恢复 checkpoint")
        )
        self.output_picker = PathPicker(mode="directory", placeholder=tr("ExperimentPage", "独立运行目录的父目录"))
        self.output_picker.set_path(settings.runs_dir)
        form.addRow(tr("ExperimentPage", "实验名称"), self.name_edit)
        form.addRow(tr("ExperimentPage", "工作流"), self.workflow_combo)
        form.addRow(tr("ExperimentPage", "已有数据集"), self.dataset_picker)
        form.addRow(tr("ExperimentPage", "单样本文件"), self.sample_picker)
        form.addRow(tr("ExperimentPage", "Checkpoint"), self.checkpoint_picker)
        form.addRow(tr("ExperimentPage", "运行目录"), self.output_picker)
        identity.body_layout.addLayout(form)
        root.addWidget(identity)

        self.generation_card = SectionCard(
            tr("ExperimentPage", "数据生成与仿真"),
            tr("ExperimentPage", "软件只读取本地 NYUv2/Middlebury 数据，不会自动下载原始数据。"),
        )
        generation_form = QFormLayout()
        self.generation_enabled = QCheckBox(tr("ExperimentPage", "在完整流程中先生成数据"))
        self.dataset_mode_combo = QComboBox()
        for label, key in (("NYUv2 raw", "raw"), ("NYUv2 labeled", "labeled"), ("Middlebury", "middlebury")):
            self.dataset_mode_combo.addItem(label, key)
        self.source_picker = PathPicker(mode="open_file", placeholder=tr("ExperimentPage", "本地原始数据路径"))
        self.simulation_combo = QComboBox()
        self.simulation_combo.setObjectName("simulationAlgorithmCombo")
        for key, spec in SIMULATION_ALGORITHMS.items():
            self.simulation_combo.addItem(tr("Algorithms", spec.label), key)
        generation_form.addRow(self.generation_enabled)
        generation_form.addRow(tr("ExperimentPage", "数据源"), self.dataset_mode_combo)
        generation_form.addRow(tr("ExperimentPage", "原始数据路径"), self.source_picker)
        generation_form.addRow(tr("ExperimentPage", "仿真模型"), self.simulation_combo)
        self.generation_card.body_layout.addLayout(generation_form)
        self.generation_state = ParameterFormState("simulation", "single")
        self.generation_parameters = DynamicParameterForm(self.generation_state)
        self.generation_card.body_layout.addWidget(self.generation_parameters)
        root.addWidget(self.generation_card)

        self.training_card = SectionCard(
            tr("ExperimentPage", "训练实验室"),
            tr("ExperimentPage", "快速预设为 1 epoch / 2 样本；标准预设使用现有 CLI 默认值。"),
        )
        training_header = QFormLayout()
        self.training_enabled = QCheckBox(tr("ExperimentPage", "在完整流程中训练模型"))
        self.training_enabled.setChecked(True)
        self.training_mode_combo = QComboBox()
        self.training_mode_combo.setObjectName("trainingModeCombo")
        self.training_mode_combo.addItem(tr("ExperimentPage", "监督重建"), "supervised")
        self.training_mode_combo.addItem(tr("ExperimentPage", "自监督 SPISR"), "self_supervised_spisr")
        self.reconstruction_combo = QComboBox()
        self.reconstruction_combo.setObjectName("reconstructionAlgorithmCombo")
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("presetCombo")
        self.preset_combo.addItem(tr("ExperimentPage", "快速"), "quick")
        self.preset_combo.addItem(tr("ExperimentPage", "标准"), "standard")
        self.preset_combo.addItem(tr("ExperimentPage", "自定义"), "custom")
        self.resume_picker = PathPicker(
            mode="open_file", file_filter="PyTorch checkpoint (*.pt *.pth)", placeholder=tr("ExperimentPage", "可选：兼容任务的断点")
        )
        self.checkpoint_hint = QLabel()
        self.checkpoint_hint.setObjectName("muted")
        self.checkpoint_hint.setWordWrap(True)
        training_header.addRow(self.training_enabled)
        training_header.addRow(tr("ExperimentPage", "训练方式"), self.training_mode_combo)
        training_header.addRow(tr("ExperimentPage", "重建算法"), self.reconstruction_combo)
        training_header.addRow(tr("ExperimentPage", "参数预设"), self.preset_combo)
        training_header.addRow(tr("ExperimentPage", "恢复训练"), self.resume_picker)
        training_header.addRow("", self.checkpoint_hint)
        self.training_card.body_layout.addLayout(training_header)
        self.training_state = ParameterFormState("reconstruction", "simple3d", preset="quick")
        self.training_parameters = DynamicParameterForm(self.training_state)
        self.training_card.body_layout.addWidget(self.training_parameters)
        root.addWidget(self.training_card)

        self.inference_card = SectionCard(
            tr("ExperimentPage", "预测、评估与画廊"),
            tr("ExperimentPage", "预测会从 checkpoint 元数据识别算法；此处不会让用户重复选择模型。"),
        )
        inference_form = QFormLayout()
        self.prediction_enabled = QCheckBox(tr("ExperimentPage", "运行预测"))
        self.prediction_enabled.setChecked(True)
        self.evaluation_enabled = QCheckBox(tr("ExperimentPage", "运行整套数据评估"))
        self.evaluation_enabled.setChecked(True)
        self.comparison_edit = QPlainTextEdit()
        self.comparison_edit.setPlaceholderText(
            tr("ExperimentPage", "多模型比较：每行填写 标签|checkpoint路径；留空则使用上方 checkpoint 或本次训练结果")
        )
        self.comparison_edit.setMaximumHeight(90)
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setObjectName("galleryCountSpin")
        self.sample_count_spin.setRange(1, 12)
        self.sample_count_spin.setValue(4)
        inference_form.addRow(self.prediction_enabled)
        inference_form.addRow(self.evaluation_enabled)
        inference_form.addRow(tr("ExperimentPage", "多 checkpoint 比较"), self.comparison_edit)
        inference_form.addRow(tr("ExperimentPage", "画廊样本数"), self.sample_count_spin)
        self.inference_card.body_layout.addLayout(inference_form)
        root.addWidget(self.inference_card)

        compute = SectionCard(
            tr("ExperimentPage", "设备与执行"),
            tr("ExperimentPage", "Auto 优先 NVIDIA；驱动或 CUDA 自检失败时 worker 会回退 CPU 并记录原因。"),
        )
        compute_form = QFormLayout()
        self.device_combo = QComboBox()
        self.device_combo.setObjectName("deviceCombo")
        for key, label in DEVICE_LABELS.items():
            self.device_combo.addItem(tr("DeviceLabels", label), key)
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(settings.device)))
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(0, 31)
        self.gpu_spin.setValue(settings.gpu_index)
        self.inspect_button = QPushButton(tr("ExperimentPage", "预览第一个样本"))
        self.inspect_button.clicked.connect(self._inspect_sample)
        compute_form.addRow(tr("ExperimentPage", "设备"), self.device_combo)
        compute_form.addRow(tr("ExperimentPage", "GPU 编号"), self.gpu_spin)
        compute_form.addRow("", self.inspect_button)
        compute.body_layout.addLayout(compute_form)
        root.addWidget(compute)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.save_button = QPushButton(tr("ExperimentPage", "保存配置"))
        self.start_button = QPushButton(tr("ExperimentPage", "开始运行"))
        self.start_button.setObjectName("primaryButton")
        actions.addWidget(self.save_button)
        actions.addWidget(self.start_button)
        root.addLayout(actions)
        root.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll_page(page))
        self.workflow_combo.currentIndexChanged.connect(self._update_workflow)
        self.dataset_mode_combo.currentIndexChanged.connect(self._update_source_mode)
        self.simulation_combo.currentIndexChanged.connect(self._simulation_changed)
        self.training_mode_combo.currentIndexChanged.connect(self._training_mode_changed)
        self.reconstruction_combo.currentIndexChanged.connect(self._reconstruction_changed)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.device_combo.currentIndexChanged.connect(
            lambda: self.gpu_spin.setEnabled(self.device_combo.currentData() != "cpu")
        )
        self.save_button.clicked.connect(self._save_config)
        self.start_button.clicked.connect(self._start)
        self._training_mode_changed()
        self._update_source_mode()
        self._update_workflow()

    def _update_workflow(self) -> None:
        workflow = self.workflow_combo.currentData()
        full = workflow == "full_pipeline"
        self.generation_card.setVisible(full or workflow == "generate")
        self.training_card.setVisible(full or workflow == "train")
        self.inference_card.setVisible(full or workflow in {"predict", "evaluate"})
        self.sample_picker.setVisible(workflow == "predict")
        self.checkpoint_picker.setVisible(workflow in {"predict", "evaluate"} or full)
        self.dataset_picker.setVisible(workflow != "predict" or full)
        self.generation_enabled.setVisible(full)
        self.training_enabled.setVisible(full)
        self.prediction_enabled.setVisible(full or workflow == "predict")
        self.evaluation_enabled.setVisible(full or workflow == "evaluate")
        if workflow == "predict":
            self.prediction_enabled.setChecked(True)
            self.evaluation_enabled.setChecked(False)
        elif workflow == "evaluate":
            self.prediction_enabled.setChecked(False)
            self.evaluation_enabled.setChecked(True)

    def _update_source_mode(self) -> None:
        mode = self.dataset_mode_combo.currentData()
        if mode == "labeled":
            self.source_picker.mode = "open_file"
            self.source_picker.file_filter = "MAT (*.mat)"
            self.source_picker.line_edit.setPlaceholderText("NYUv2 .mat")
        else:
            self.source_picker.mode = "directory"
            self.source_picker.line_edit.setPlaceholderText(
                "NYUv2 raw 根目录" if mode == "raw" else "Middlebury raw 根目录"
            )

    def _simulation_changed(self) -> None:
        key = self.simulation_combo.currentData()
        if key:
            self.generation_state.set_algorithm(key)
            self.generation_parameters.set_state(self.generation_state)

    def _training_mode_changed(self) -> None:
        family = self.training_mode_combo.currentData()
        prior = self.reconstruction_combo.currentData()
        self.reconstruction_combo.blockSignals(True)
        self.reconstruction_combo.clear()
        for key, spec in RECONSTRUCTION_ALGORITHMS.items():
            if spec.method_family == family:
                self.reconstruction_combo.addItem(tr("Algorithms", spec.label), key)
        selected = self.reconstruction_combo.findData(prior)
        self.reconstruction_combo.setCurrentIndex(selected if selected >= 0 else 0)
        self.reconstruction_combo.blockSignals(False)
        self._reconstruction_changed()

    def _reconstruction_changed(self) -> None:
        key = self.reconstruction_combo.currentData()
        if not key:
            return
        self.training_state.set_algorithm(key)
        self.training_parameters.set_state(self.training_state)
        spec = RECONSTRUCTION_ALGORITHMS[key]
        self.checkpoint_hint.setText(
            tr("ExperimentPage", "随软件提供预训练 checkpoint，可直接预测。")
            if spec.bundled_checkpoint
            else tr("ExperimentPage", "该算法不内置 checkpoint，需要先训练或导入兼容 checkpoint。")
        )

    def _preset_changed(self) -> None:
        preset = self.preset_combo.currentData()
        if preset:
            self.training_state.apply_preset(preset)
            self.training_parameters.set_state(self.training_state)

    def _parse_comparisons(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        labels: list[str] = []
        checkpoints: list[str] = []
        for line_number, raw in enumerate(self.comparison_edit.toPlainText().splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            if "|" not in line:
                raise ValueError(f"多模型比较第 {line_number} 行需要使用 标签|路径")
            label, path = (part.strip() for part in line.split("|", 1))
            if not label or not path:
                raise ValueError(f"多模型比较第 {line_number} 行不完整")
            labels.append(label)
            checkpoints.append(path)
        return tuple(labels), tuple(checkpoints)

    def build_config(self) -> RunConfig:
        display_name = self.name_edit.text().strip()
        if not display_name:
            raise ValueError("实验名称不能为空")
        output_root = self.output_picker.path()
        if not output_root:
            raise ValueError("请选择运行目录")
        labels, comparisons = self._parse_comparisons()
        dataset = self.dataset_picker.path()
        sample = self.sample_picker.path()
        checkpoint = self.checkpoint_picker.path()
        workflow = self.workflow_combo.currentData()
        source = self.source_picker.path()
        request = ExperimentRequest(
            workflow=workflow,
            display_name=display_name,
            dataset_paths=(dataset,) if dataset else (),
            source_path=source or None,
            sample_file=sample or None,
            checkpoint_paths=(checkpoint,) if checkpoint else (),
            generation_enabled=self.generation_enabled.isChecked(),
            dataset_mode=self.dataset_mode_combo.currentData(),
            simulation_model=self.simulation_combo.currentData(),
            generation_parameters=self.generation_parameters.values(),
            training_enabled=self.training_enabled.isChecked(),
            reconstruction_model=self.reconstruction_combo.currentData(),
            training_preset=self.preset_combo.currentData(),
            training_parameters=self.training_parameters.values(),
            resume_checkpoint=self.resume_picker.path() or None,
            prediction_enabled=self.prediction_enabled.isChecked(),
            prediction_checkpoint=checkpoint or None,
            evaluation_enabled=self.evaluation_enabled.isChecked(),
            evaluation_checkpoints=comparisons,
            evaluation_labels=labels,
            sample_count=self.sample_count_spin.value(),
            device=self.device_combo.currentData(),
            gpu_index=self.gpu_spin.value(),
        )
        run_dir = next_run_directory(output_root, display_name)
        return request.to_run_config(run_dir, history_db=self.paths.history_db)

    def load_config(self, config: RunConfig) -> None:
        self.name_edit.setText(config.display_name + tr("ExperimentPage", "（副本）"))
        index = self.workflow_combo.findData(config.workflow)
        if index >= 0 and config.workflow not in {"quick_demo", "noop"}:
            self.workflow_combo.setCurrentIndex(index)
        self.dataset_picker.set_path(config.input.dataset_paths[0] if config.input.dataset_paths else "")
        self.source_picker.set_path(config.input.source_path)
        self.sample_picker.set_path(config.prediction.sample_file or config.input.sample_file)
        checkpoint = config.prediction.checkpoint or (config.input.checkpoint_paths[0] if config.input.checkpoint_paths else None)
        self.checkpoint_picker.set_path(checkpoint)
        self.output_picker.set_path(Path(config.output.run_dir).parent)
        mode_index = self.dataset_mode_combo.findData(config.generation.dataset_mode)
        self.dataset_mode_combo.setCurrentIndex(max(0, mode_index))
        simulation_index = self.simulation_combo.findData(config.generation.surface_model)
        self.simulation_combo.setCurrentIndex(max(0, simulation_index))
        self.generation_state.values.update(config.generation.parameters)
        self.generation_parameters.set_state(self.generation_state)
        family = RECONSTRUCTION_ALGORITHMS[config.training.model].method_family
        family_index = self.training_mode_combo.findData(family)
        self.training_mode_combo.setCurrentIndex(max(0, family_index))
        model_index = self.reconstruction_combo.findData(config.training.model)
        self.reconstruction_combo.setCurrentIndex(max(0, model_index))
        preset_index = self.preset_combo.findData(config.training.preset)
        self.preset_combo.setCurrentIndex(max(0, preset_index))
        self.training_state.values.update(config.training.parameters)
        self.training_parameters.set_state(self.training_state)
        self.resume_picker.set_path(config.training.resume_checkpoint)
        self.generation_enabled.setChecked(config.generation.enabled)
        self.training_enabled.setChecked(config.training.enabled)
        self.prediction_enabled.setChecked(config.prediction.enabled)
        self.evaluation_enabled.setChecked(config.evaluation.enabled)
        self.sample_count_spin.setValue(config.visualization.sample_count)
        device_index = self.device_combo.findData(config.compute.preference)
        self.device_combo.setCurrentIndex(max(0, device_index))
        self.gpu_spin.setValue(config.compute.gpu_index)
        labels = config.evaluation.labels or tuple(Path(path).stem for path in config.evaluation.checkpoints)
        lines = [f"{label}|{path}" for label, path in zip(labels, config.evaluation.checkpoints)]
        self.comparison_edit.setPlainText("\n".join(lines))

    def _show_validation_error(self, exc: Exception) -> None:
        QMessageBox.warning(self, tr("ExperimentPage", "配置需要调整"), str(exc))

    def _save_config(self) -> None:
        try:
            config = self.build_config()
            run_dir = Path(config.output.run_dir)
            run_dir.mkdir(parents=True, exist_ok=False)
            atomic_write_text(run_dir / "run.json", config.to_json())
            self._last_config = config
            self.config_saved.emit(str(run_dir / "run.json"))
        except (OSError, ValueError) as exc:
            self._show_validation_error(exc)

    def _start(self) -> None:
        try:
            config = self.build_config()
        except ValueError as exc:
            self._show_validation_error(exc)
            return
        self._last_config = config
        self.run_requested.emit(config)

    def _inspect_sample(self) -> None:
        sample = self.sample_picker.path()
        if not sample:
            dataset = Path(self.dataset_picker.path()).expanduser() if self.dataset_picker.path() else None
            if dataset and dataset.is_dir():
                samples = sorted(dataset.glob("sample_*.npz")) or sorted(dataset.glob("*.npz"))
                sample = str(samples[0]) if samples else ""
        if not sample:
            QMessageBox.information(self, tr("ExperimentPage", "没有样本"), tr("ExperimentPage", "请先选择 NPZ 样本或数据集。"))
            return
        try:
            dialog = SampleInspectorDialog(sample, self)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, tr("ExperimentPage", "无法读取样本"), str(exc))
            return
        dialog.exec()


class RunPage(QWidget):
    open_results_requested = Signal(str)
    resume_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller: WorkerController | None = None
        self.run_dir: str = ""
        self._close_after_completion = False
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 24, 26, 26)
        root.setSpacing(14)
        root.addWidget(
            PageHeader(
                tr("RunPage", "运行监控"),
                tr("RunPage", "worker 在独立进程中执行；关闭应用前会先请求安全取消。"),
            )
        )
        status_card = SectionCard(tr("RunPage", "任务状态"))
        status_row = QHBoxLayout()
        self.run_name = QLabel(tr("RunPage", "尚未开始"))
        self.run_name.setObjectName("cardTitle")
        self.status_badge = StatusBadge("preparing")
        self.stage_label = QLabel("")
        self.stage_label.setObjectName("muted")
        status_row.addWidget(self.run_name)
        status_row.addStretch(1)
        status_row.addWidget(self.status_badge)
        status_card.body_layout.addLayout(status_row)
        status_card.body_layout.addWidget(self.stage_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_card.body_layout.addWidget(self.progress_bar)
        controls = QHBoxLayout()
        self.cancel_button = QPushButton(tr("RunPage", "安全取消"))
        self.cancel_button.setObjectName("dangerButton")
        self.cancel_button.setEnabled(False)
        self.result_button = QPushButton(tr("RunPage", "打开结果"))
        self.result_button.setEnabled(False)
        self.resume_button = QPushButton(tr("RunPage", "恢复任务"))
        self.resume_button.setEnabled(False)
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        controls.addWidget(self.resume_button)
        controls.addWidget(self.result_button)
        status_card.body_layout.addLayout(controls)
        root.addWidget(status_card)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        charts = SectionCard(tr("RunPage", "实时曲线"), tr("RunPage", "显示 worker 发出的 loss / MAE / RMSE 事件。"))
        self.loss_chart = LineChartWidget("Loss")
        self.metric_chart = LineChartWidget("MAE / RMSE")
        charts.body_layout.addWidget(self.loss_chart)
        charts.body_layout.addWidget(self.metric_chart)
        log_card = SectionCard(tr("RunPage", "运行日志"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        log_card.body_layout.addWidget(self.log_view)
        splitter.addWidget(charts)
        splitter.addWidget(log_card)
        splitter.setSizes([520, 520])
        root.addWidget(splitter, 1)
        self.cancel_button.clicked.connect(self._cancel)
        self.result_button.clicked.connect(lambda: self.open_results_requested.emit(self.run_dir))
        self.resume_button.clicked.connect(lambda: self.resume_requested.emit(self.run_dir))

    def bind_controller(self, controller: WorkerController) -> None:
        if self.controller is controller:
            return
        self.controller = controller
        controller.progress_changed.connect(self._update_progress)
        controller.log_received.connect(self._append_log)
        controller.protocol_warning.connect(lambda message: self._append_log("[提示] " + message))
        controller.launch_failed.connect(lambda message: self._append_log("[启动失败] " + message))
        controller.completed.connect(self._completed)

    def start_run(self, config: RunConfig) -> None:
        if self.controller is None:
            raise RuntimeError("RunPage 尚未绑定 WorkerController")
        self.run_dir = config.output.run_dir
        self.run_name.setText(config.display_name)
        self.status_badge.set_status("preparing")
        self.stage_label.setText(
            tr("WorkflowLabels", WORKFLOW_LABELS.get(config.workflow, config.workflow))
        )
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.cancel_button.setEnabled(True)
        self.result_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.controller.start(config)

    def _update_progress(self, state: RunProgressState) -> None:
        self.status_badge.set_status(state.status)
        detail = state.stage
        if state.total:
            detail += f" · {state.current}/{state.total}"
        self.stage_label.setText(
            detail or tr("StatusLabels", STATUS_LABELS.get(state.status, state.status))
        )
        self.progress_bar.setValue(state.percent)
        loss = {key: points for key, points in state.metrics.items() if "loss" in key.lower()}
        metrics = {key: points for key, points in state.metrics.items() if any(token in key.lower() for token in ("mae", "rmse", "absrel"))}
        self.loss_chart.set_series(loss)
        self.metric_chart.set_series(metrics)

    def _append_log(self, message: str) -> None:
        if message:
            self.log_view.appendPlainText(message)

    def _cancel(self) -> None:
        if self.controller and self.controller.request_cancel():
            self.cancel_button.setEnabled(False)
            self.stage_label.setText(tr("RunPage", "正在保存安全状态并取消…"))

    def _completed(self, status: str, exit_code: int) -> None:
        self.status_badge.set_status(status)
        self.cancel_button.setEnabled(False)
        self.result_button.setEnabled(bool(self.run_dir and Path(self.run_dir, "result_manifest.json").is_file()))
        can_resume = False
        if status in RESUMABLE_STATUSES and self.run_dir:
            try:
                can_resume = ResultGalleryModel.load(self.run_dir).resume_available()
            except ValueError:
                pass
        self.resume_button.setEnabled(can_resume)
        status_label = tr("StatusLabels", STATUS_LABELS.get(status, status))
        self._append_log(f"[{status_label}] worker exit code: {exit_code}")


class ResultsPage(QWidget):
    reuse_config_requested = Signal(object)
    resume_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model: ResultGalleryModel | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 24, 26, 26)
        root.setSpacing(12)
        header_row = QHBoxLayout()
        header_row.addWidget(PageHeader(tr("ResultsPage", "结果画廊"), tr("ResultsPage", "有标签时显示目标与误差；无标签时自动隐藏监督指标。")), 1)
        self.open_button = QPushButton(tr("ResultsPage", "打开结果…"))
        self.open_button.clicked.connect(self.choose_result)
        header_row.addWidget(self.open_button)
        root.addLayout(header_row)

        summary = SectionCard(tr("ResultsPage", "运行摘要"))
        summary_row = QHBoxLayout()
        self.title_label = QLabel(tr("ResultsPage", "尚未打开结果"))
        self.title_label.setObjectName("cardTitle")
        self.status_badge = StatusBadge("preparing")
        self.count_spin = QSpinBox()
        self.count_spin.setObjectName("resultsGalleryCountSpin")
        self.count_spin.setRange(1, 12)
        self.count_spin.setValue(4)
        self.count_spin.valueChanged.connect(self.refresh_gallery)
        self.reuse_button = QPushButton(tr("ResultsPage", "复用配置"))
        self.resume_button = QPushButton(tr("ResultsPage", "恢复任务"))
        self.export_button = QPushButton(tr("ResultsPage", "导出结果…"))
        self.reuse_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.reuse_button.clicked.connect(self._reuse)
        self.resume_button.clicked.connect(self._resume)
        self.export_button.clicked.connect(self._export)
        summary_row.addWidget(self.title_label)
        summary_row.addWidget(self.status_badge)
        summary_row.addStretch(1)
        summary_row.addWidget(QLabel(tr("ResultsPage", "样本数")))
        summary_row.addWidget(self.count_spin)
        summary_row.addWidget(self.reuse_button)
        summary_row.addWidget(self.resume_button)
        summary_row.addWidget(self.export_button)
        summary.body_layout.addLayout(summary_row)
        root.addWidget(summary)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        metrics_card = SectionCard(tr("ResultsPage", "指标"))
        self.metrics_tree = QTreeWidget()
        self.metrics_tree.setHeaderLabels([tr("ResultsPage", "指标"), tr("ResultsPage", "值")])
        metrics_card.body_layout.addWidget(self.metrics_tree)
        artifacts_card = SectionCard(tr("ResultsPage", "产物"), tr("ResultsPage", "双击在系统默认程序中打开。"))
        self.artifact_table = QTableWidget(0, 3)
        self.artifact_table.setHorizontalHeaderLabels([tr("ResultsPage", "名称"), tr("ResultsPage", "类型"), tr("ResultsPage", "路径")])
        self.artifact_table.horizontalHeader().setStretchLastSection(True)
        self.artifact_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.artifact_table.doubleClicked.connect(self._open_artifact)
        artifacts_card.body_layout.addWidget(self.artifact_table)
        splitter.addWidget(metrics_card)
        splitter.addWidget(artifacts_card)
        splitter.setSizes([380, 650])
        root.addWidget(splitter)

        self.gallery_content = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_content)
        self.gallery_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_layout.setSpacing(12)
        gallery_scroll = _scroll_page(self.gallery_content)
        root.addWidget(gallery_scroll, 1)

    def choose_result(self) -> bool:
        selected = QFileDialog.getExistingDirectory(self, tr("ResultsPage", "选择运行目录"))
        if selected:
            try:
                self.load_result(selected)
            except ValueError as exc:
                QMessageBox.warning(self, tr("ResultsPage", "无法打开结果"), str(exc))
                return False
            return True
        return False

    def load_result(self, value: str | Path) -> None:
        self.model = ResultGalleryModel.load(value)
        self.title_label.setText(self.model.config.display_name)
        self.status_badge.set_status(self.model.manifest.status)
        self.count_spin.setValue(self.model.config.visualization.sample_count)
        self.reuse_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.resume_button.setEnabled(self.model.resume_available())
        self._populate_artifacts()
        self.refresh_gallery()

    def _populate_artifacts(self) -> None:
        assert self.model is not None
        artifacts = self.model.artifacts
        self.artifact_table.setRowCount(len(artifacts))
        for row, (record, path) in enumerate(artifacts):
            for column, value in enumerate((record.name, record.kind, record.path)):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.artifact_table.setItem(row, column, item)

    def refresh_gallery(self) -> None:
        _clear_layout(self.gallery_layout)
        self.metrics_tree.clear()
        if self.model is None:
            empty = QLabel(tr("ResultsPage", "打开一个运行目录后，这里会显示结果。"))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.gallery_layout.addWidget(empty)
            return
        indices = self.model.selected_indices(self.count_spin.value())
        samples = [self.model.load_sample(index) for index in indices]
        labeled = any(sample.labeled for sample in samples)
        self._add_metrics(self.model.filtered_metrics(labeled=labeled))
        if not samples:
            empty = QLabel(tr("ResultsPage", "该结果清单中没有可展示的样本产物。"))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.gallery_layout.addWidget(empty)
        for sample in samples:
            self.gallery_layout.addWidget(GalleryCard(sample))
        self.gallery_layout.addStretch(1)

    def _add_metrics(self, metrics: dict[str, Any], parent: QTreeWidgetItem | None = None) -> None:
        for key, value in metrics.items():
            item = QTreeWidgetItem([str(key), "" if isinstance(value, dict) else _format_value(value)])
            (parent.addChild(item) if parent is not None else self.metrics_tree.addTopLevelItem(item))
            if isinstance(value, dict):
                self._add_metrics(value, item)
        self.metrics_tree.expandAll()

    def _open_artifact(self) -> None:
        row = self.artifact_table.currentRow()
        item = self.artifact_table.item(row, 0) if row >= 0 else None
        path = item.data(Qt.ItemDataRole.UserRole) if item else None
        if path and Path(path).exists():
            _open_local_path(path)

    def _reuse(self) -> None:
        if self.model:
            self.reuse_config_requested.emit(self.model.config)

    def _resume(self) -> None:
        if self.model:
            self.resume_requested.emit(str(self.model.run_dir))

    def _export(self) -> None:
        if self.model is None:
            return
        parent = QFileDialog.getExistingDirectory(self, tr("ResultsPage", "选择导出位置"))
        if not parent:
            return
        destination = Path(parent) / f"SPImaging-export-{self.model.config.run_id[:8]}"
        try:
            destination.mkdir(parents=True, exist_ok=False)
            allowed = {".csv", ".json", ".jsonl", ".png", ".jpg", ".jpeg", ".npz", ".log"}
            for _record, source in self.model.artifacts:
                if source.is_file() and source.suffix.lower() in allowed:
                    target = destination / source.name
                    if target.exists():
                        target = destination / f"{source.parent.name}_{source.name}"
                    shutil.copy2(source, target)
            for name in ("run.json", "result_manifest.json"):
                source = self.model.run_dir / name
                if source.is_file():
                    shutil.copy2(source, destination / name)
            QMessageBox.information(self, tr("ResultsPage", "导出完成"), str(destination))
        except OSError as exc:
            QMessageBox.warning(self, tr("ResultsPage", "导出失败"), str(exc))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


class HistoryPage(QWidget):
    open_requested = Signal(str)
    reuse_requested = Signal(str)
    resume_requested = Signal(str)

    def __init__(self, model: RunHistoryModel, runs_root: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.runs_root = Path(runs_root)
        self.records: list[HistoryRecord] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 24, 26, 26)
        root.setSpacing(12)
        header = QHBoxLayout()
        header.addWidget(PageHeader(tr("HistoryPage", "历史与恢复"), tr("HistoryPage", "SQLite 仅作索引；可随时从独立运行目录重建。")), 1)
        self.refresh_button = QPushButton(tr("HistoryPage", "刷新"))
        self.rebuild_button = QPushButton(tr("HistoryPage", "重建索引"))
        header.addWidget(self.refresh_button)
        header.addWidget(self.rebuild_button)
        root.addLayout(header)
        card = SectionCard(tr("HistoryPage", "全部运行"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            [tr("HistoryPage", "名称"), tr("HistoryPage", "工作流"), tr("HistoryPage", "状态"), tr("HistoryPage", "创建时间"), tr("HistoryPage", "运行目录")]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(self._open)
        card.body_layout.addWidget(self.table)
        actions = QHBoxLayout()
        self.open_button = QPushButton(tr("HistoryPage", "打开结果"))
        self.reuse_button = QPushButton(tr("HistoryPage", "复用配置"))
        self.resume_button = QPushButton(tr("HistoryPage", "恢复任务"))
        for button in (self.open_button, self.reuse_button, self.resume_button):
            button.setEnabled(False)
        actions.addStretch(1)
        actions.addWidget(self.open_button)
        actions.addWidget(self.reuse_button)
        actions.addWidget(self.resume_button)
        card.body_layout.addLayout(actions)
        root.addWidget(card, 1)
        self.refresh_button.clicked.connect(self.refresh)
        self.rebuild_button.clicked.connect(self.rebuild)
        self.open_button.clicked.connect(self._open)
        self.reuse_button.clicked.connect(self._reuse)
        self.resume_button.clicked.connect(self._resume)

    def refresh(self) -> None:
        self.records = self.model.list(limit=250)
        self.table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            values = (
                record.display_name,
                tr("WorkflowLabels", WORKFLOW_LABELS.get(record.workflow, record.workflow)),
                tr("StatusLabels", STATUS_LABELS.get(record.status, record.status)),
                record.created_at,
                record.run_dir,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, record.run_dir)
                self.table.setItem(row, column, item)
        self._selection_changed()

    def rebuild(self) -> None:
        root = QFileDialog.getExistingDirectory(self, tr("HistoryPage", "选择运行目录根路径"), str(self.runs_root))
        if not root:
            return
        imported, errors = self.model.rebuild((root,))
        message = tr("HistoryPage", "已导入 {count} 条记录。 " ).format(count=imported)
        if errors:
            message += tr("HistoryPage", "有 {count} 个目录无法读取。" ).format(count=len(errors))
        QMessageBox.information(self, tr("HistoryPage", "索引重建完成"), message)
        self.refresh()

    def _selected(self) -> HistoryRecord | None:
        row = self.table.currentRow()
        return self.records[row] if 0 <= row < len(self.records) else None

    def _selection_changed(self) -> None:
        record = self._selected()
        self.open_button.setEnabled(record is not None)
        self.reuse_button.setEnabled(record is not None and Path(record.run_dir, "run.json").is_file())
        can_resume = False
        if record is not None and record.status in RESUMABLE_STATUSES:
            try:
                can_resume = ResultGalleryModel.load(record.run_dir).resume_available()
            except ValueError:
                pass
        self.resume_button.setEnabled(can_resume)

    def _open(self) -> None:
        record = self._selected()
        if record:
            self.open_requested.emit(record.run_dir)

    def _reuse(self) -> None:
        record = self._selected()
        if record:
            self.reuse_requested.emit(record.run_dir)

    def _resume(self) -> None:
        record = self._selected()
        if record:
            self.resume_requested.emit(record.run_dir)


class DeviceProbeThread(QThread):
    result_ready = Signal(str, bool)

    def __init__(self, mode: str, gpu_index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode = mode
        self.gpu_index = gpu_index

    def run(self) -> None:
        try:
            from spimaging.training_common.device import select_torch_device

            result = select_torch_device(self.mode, self.gpu_index)
            details = str(result.reason)
            try:
                import torch

                if getattr(result.device, "type", "cpu") == "cuda":
                    free, total = torch.cuda.mem_get_info(self.gpu_index)
                    details += f" 可用显存 {free / 1024**3:.1f} / {total / 1024**3:.1f} GiB。"
                else:
                    from spimaging.training_common.security import available_memory_bytes

                    available = available_memory_bytes()
                    if available is not None:
                        details += f" 当前可用内存约 {available / 1024**3:.1f} GiB。"
            except Exception as exc:
                details += f" 内存预检未完成（{exc}）。"
            self.result_ready.emit(details, bool(result.fallback))
        except Exception as exc:
            self.result_ready.emit(f"设备检测失败：{exc}", True)


class SettingsPage(QWidget):
    settings_saved = Signal(object)
    environment_repair_requested = Signal()

    def __init__(self, store: SettingsStore, settings: DesktopSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self._probe: DeviceProbeThread | None = None
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(26, 24, 26, 26)
        root.setSpacing(14)
        root.addWidget(PageHeader(tr("SettingsPage", "设置"), tr("SettingsPage", "运行时、设备、缓存、更新与本地诊断。不会上传用户数据。")))

        device = SectionCard(tr("SettingsPage", "计算设备"))
        device_form = QFormLayout()
        self.device_combo = QComboBox()
        for key, label in DEVICE_LABELS.items():
            self.device_combo.addItem(tr("DeviceLabels", label), key)
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(settings.device)))
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(0, 31)
        self.gpu_spin.setValue(settings.gpu_index)
        self.probe_button = QPushButton(tr("SettingsPage", "运行设备与显存预检"))
        self.probe_result = QLabel(tr("SettingsPage", "尚未检测"))
        self.probe_result.setObjectName("muted")
        self.probe_result.setWordWrap(True)
        device_form.addRow(tr("SettingsPage", "偏好"), self.device_combo)
        device_form.addRow(tr("SettingsPage", "GPU 编号"), self.gpu_spin)
        device_form.addRow("", self.probe_button)
        device_form.addRow(tr("SettingsPage", "检测结果"), self.probe_result)
        device.body_layout.addLayout(device_form)
        root.addWidget(device)

        storage = SectionCard(tr("SettingsPage", "存储与缓存"))
        storage_form = QFormLayout()
        self.runs_picker = PathPicker(mode="directory")
        self.runs_picker.set_path(settings.runs_dir)
        self.cache_picker = PathPicker(mode="directory")
        self.cache_picker.set_path(settings.cache_dir)
        storage_form.addRow(tr("SettingsPage", "运行目录"), self.runs_picker)
        storage_form.addRow(tr("SettingsPage", "缓存目录"), self.cache_picker)
        storage.body_layout.addLayout(storage_form)
        root.addWidget(storage)

        maintenance = SectionCard(
            tr("SettingsPage", "更新、修复与诊断"),
            tr("SettingsPage", "公开测试版仅在用户确认后安装更新；任务运行时不会更新。"),
        )
        self.update_check = QCheckBox(tr("SettingsPage", "每 24 小时最多自动检查一次更新"))
        self.update_check.setChecked(settings.update_checks)
        self.repair_button = QPushButton(tr("SettingsPage", "修复私有运行环境"))
        self.diagnostic_button = QPushButton(tr("SettingsPage", "导出脱敏诊断包…"))
        row = QHBoxLayout()
        row.addWidget(self.repair_button)
        row.addWidget(self.diagnostic_button)
        row.addStretch(1)
        maintenance.body_layout.addWidget(self.update_check)
        maintenance.body_layout.addLayout(row)
        root.addWidget(maintenance)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.save_button = QPushButton(tr("SettingsPage", "保存设置"))
        self.save_button.setObjectName("primaryButton")
        actions.addWidget(self.save_button)
        root.addLayout(actions)
        root.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll_page(page))
        self.probe_button.clicked.connect(self._probe_device)
        self.repair_button.clicked.connect(self.environment_repair_requested)
        self.diagnostic_button.clicked.connect(self._export_diagnostics)
        self.save_button.clicked.connect(self._save)

    def current_settings(self) -> DesktopSettings:
        return DesktopSettings(
            device=self.device_combo.currentData(),
            gpu_index=self.gpu_spin.value(),
            runs_dir=self.runs_picker.path(),
            cache_dir=self.cache_picker.path(),
            update_checks=self.update_check.isChecked(),
            locale="zh_CN",
        )

    def _save(self) -> None:
        try:
            settings = self.current_settings()
            self.store.save(settings)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, tr("SettingsPage", "保存失败"), str(exc))
            return
        self.settings_saved.emit(settings)
        QMessageBox.information(self, tr("SettingsPage", "设置已保存"), tr("SettingsPage", "新的运行将使用这些设置。"))

    def _probe_device(self) -> None:
        if self._probe is not None and self._probe.isRunning():
            return
        self.probe_button.setEnabled(False)
        self.probe_result.setText(tr("SettingsPage", "正在检测 CUDA 与显存可用性…"))
        self._probe = DeviceProbeThread(self.device_combo.currentData(), self.gpu_spin.value(), self)
        self._probe.result_ready.connect(self._probe_finished)
        self._probe.finished.connect(lambda: self.probe_button.setEnabled(True))
        self._probe.start()

    def _probe_finished(self, reason: str, fallback: bool) -> None:
        prefix = tr("SettingsPage", "已回退 CPU：") if fallback else tr("SettingsPage", "设备可用：")
        self.probe_result.setText(prefix + reason)

    def _export_diagnostics(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            tr("SettingsPage", "导出诊断包"),
            str(Path.home() / "SPImaging-diagnostics.zip"),
            "ZIP (*.zip)",
        )
        if not selected:
            return
        try:
            result = export_diagnostic_bundle(selected)
            QMessageBox.information(self, tr("SettingsPage", "诊断包已导出"), str(result))
        except OSError as exc:
            QMessageBox.warning(self, tr("SettingsPage", "导出失败"), str(exc))


__all__ = [
    "ExperimentPage",
    "HistoryPage",
    "HomePage",
    "ResultsPage",
    "RunPage",
    "SettingsPage",
]
