"""Reusable widgets for registry forms, sample inspection and result galleries."""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Any, Mapping

from spimaging.appcore.specs import ParameterSpec, ParameterType
from spimaging.desktop.dependency import require_pyside6
from spimaging.desktop.i18n import tr
from spimaging.desktop.models import (
    GallerySample,
    ParameterFormState,
    STATUS_LABELS,
    load_sample_inspection,
)


require_pyside6()

from PySide6.QtCore import Qt, Signal  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class SectionCard(QFrame):
    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.outer_layout = QVBoxLayout(self)
        self.outer_layout.setContentsMargins(18, 16, 18, 18)
        self.outer_layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        self.outer_layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("muted")
            subtitle_label.setWordWrap(True)
            self.outer_layout.addWidget(subtitle_label)
        self.body = QWidget(self)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 4, 0, 0)
        self.body_layout.setSpacing(9)
        self.outer_layout.addWidget(self.body)


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(3)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        description = QLabel(subtitle)
        description.setObjectName("pageSubtitle")
        description.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(description)


class PathPicker(QWidget):
    path_changed = Signal(str)

    def __init__(
        self,
        *,
        mode: str = "directory",
        file_filter: str = "所有文件 (*)",
        placeholder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if mode not in {"directory", "open_file", "save_file"}:
            raise ValueError(f"未知路径选择模式：{mode}")
        self.mode = mode
        self.file_filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.line_edit = QLineEdit()
        self.line_edit.setObjectName("pathLineEdit")
        self.line_edit.setPlaceholderText(placeholder)
        self.browse_button = QPushButton(tr("PathPicker", "浏览…"))
        self.browse_button.setObjectName("browseButton")
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)
        self.line_edit.textChanged.connect(self.path_changed)
        self.browse_button.clicked.connect(self._browse)

    def path(self) -> str:
        return self.line_edit.text().strip()

    def set_path(self, value: str | Path | None) -> None:
        self.line_edit.setText("" if value is None else str(value))

    def _browse(self) -> None:
        start = self.path() or str(Path.home())
        if self.mode == "directory":
            selected = QFileDialog.getExistingDirectory(self, tr("PathPicker", "选择目录"), start)
        elif self.mode == "save_file":
            selected, _ = QFileDialog.getSaveFileName(
                self, tr("PathPicker", "保存文件"), start, self.file_filter
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self, tr("PathPicker", "选择文件"), start, self.file_filter
            )
        if selected:
            self.set_path(selected)


class DynamicParameterForm(QWidget):
    """Qt editor generated solely from ``ParameterSpec`` metadata."""

    values_changed = Signal(dict)
    validation_failed = Signal(str)

    def __init__(self, state: ParameterFormState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.controls: dict[str, QWidget] = {}
        self._rows: dict[str, tuple[QLabel, QWidget, bool]] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(7)
        self.basic_widget = QWidget()
        self.basic_form = QFormLayout(self.basic_widget)
        self.basic_form.setContentsMargins(0, 0, 0, 0)
        self.basic_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        root.addWidget(self.basic_widget)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText(tr("DynamicParameterForm", "高级参数"))
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        root.addWidget(self.advanced_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.advanced_group = QGroupBox(tr("DynamicParameterForm", "高级参数"))
        self.advanced_form = QFormLayout(self.advanced_group)
        self.advanced_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.advanced_group.setVisible(False)
        root.addWidget(self.advanced_group)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.rebuild()

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.advanced_group.setVisible(checked)

    def set_state(self, state: ParameterFormState) -> None:
        self.state = state
        self.rebuild()

    def rebuild(self) -> None:
        while self.basic_form.rowCount():
            self.basic_form.removeRow(0)
        while self.advanced_form.rowCount():
            self.advanced_form.removeRow(0)
        self.controls.clear()
        self._rows.clear()
        for spec in self.state.specs:
            control = self._create_control(spec, self.state.values.get(spec.name, spec.default))
            label = QLabel(tr("Parameters", spec.label))
            if spec.help:
                help_text = tr("ParameterHelp", spec.help)
                label.setToolTip(help_text)
                control.setToolTip(help_text)
            form = self.advanced_form if spec.advanced else self.basic_form
            form.addRow(label, control)
            self.controls[spec.name] = control
            self._rows[spec.name] = (label, control, spec.advanced)
        self._update_visibility()

    def _create_control(self, spec: ParameterSpec, value: Any) -> QWidget:
        if spec.kind is ParameterType.BOOLEAN:
            control = QCheckBox()
            control.setChecked(bool(value))
            control.toggled.connect(self._control_changed)
            return control
        if spec.kind is ParameterType.INTEGER:
            control = QSpinBox()
            # QSpinBox uses a signed 32-bit Qt ``int`` even though RunConfig can
            # validate wider values (for example the uint32 seed ceiling).
            minimum = int(spec.minimum if spec.minimum is not None else -2_147_483_647)
            maximum = int(spec.maximum if spec.maximum is not None else 2_147_483_647)
            control.setRange(
                max(-2_147_483_647, minimum),
                min(2_147_483_647, maximum),
            )
            if spec.odd:
                control.setSingleStep(2)
            control.setValue(int(value))
            control.valueChanged.connect(self._control_changed)
            return control
        if spec.kind is ParameterType.NUMBER:
            control = QDoubleSpinBox()
            control.setDecimals(15)
            minimum = float(spec.minimum if spec.minimum is not None else -1e100)
            maximum = float(spec.maximum if spec.maximum is not None else 1e100)
            numeric = float(0.0 if value is None else value)
            step = 10 ** max(-12, math.floor(math.log10(abs(numeric))) - 1) if numeric else 0.01
            if spec.nullable:
                # Put the special null value exactly one editor step below the
                # valid range so one increment reaches the real minimum.
                sentinel = minimum - step
                if not math.isfinite(sentinel) or sentinel < -1e100:
                    raise ValueError(f"可空参数缺少可表示的最小值：{spec.name}")
                control.setRange(sentinel, maximum)
                control.setSpecialValueText(tr("DynamicParameterForm", "使用介质默认值"))
                control.setProperty("spimagingNullableSentinel", sentinel)
            else:
                control.setRange(minimum, maximum)
            control.setSingleStep(step)
            if value is None:
                control.setValue(float(control.property("spimagingNullableSentinel")))
            else:
                control.setValue(numeric)
            control.valueChanged.connect(self._control_changed)
            return control
        if spec.kind is ParameterType.CHOICE:
            control = QComboBox()
            control.addItems(list(spec.choices))
            control.setCurrentText(str(value))
            control.currentTextChanged.connect(self._control_changed)
            return control
        control = QLineEdit(str(value))
        control.textChanged.connect(self._control_changed)
        return control

    def _control_value(self, name: str) -> Any:
        control = self.controls[name]
        if isinstance(control, QCheckBox):
            return control.isChecked()
        if isinstance(control, (QSpinBox, QDoubleSpinBox)):
            if isinstance(control, QDoubleSpinBox):
                sentinel = control.property("spimagingNullableSentinel")
                if sentinel is not None and control.value() == float(sentinel):
                    return None
            return control.value()
        if isinstance(control, QComboBox):
            return control.currentText()
        if isinstance(control, QLineEdit):
            return control.text()
        raise TypeError(f"无法读取参数控件：{name}")

    def _control_changed(self, _value: Any = None) -> None:
        for name in self.controls:
            self.state.values[name] = self._control_value(name)
        self._update_visibility()
        try:
            resolved = self.state.resolved_values()
        except ValueError as exc:
            self.validation_failed.emit(str(exc))
            return
        self.values_changed.emit(resolved)

    def _update_visibility(self) -> None:
        values = {name: self._control_value(name) for name in self.controls}
        specs = {item.name: item for item in self.state.specs}
        for name, (label, control, _advanced) in self._rows.items():
            visible = specs[name].is_visible(values)
            label.setVisible(visible)
            control.setVisible(visible)

    def values(self) -> dict[str, Any]:
        for name in self.controls:
            self.state.values[name] = self._control_value(name)
        return self.state.resolved_values()


class StatusBadge(QLabel):
    def __init__(self, status: str = "preparing", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusBadge")
        self.set_status(status)

    def set_status(self, status: str) -> None:
        self.setProperty("status", status)
        self.setText(tr("StatusLabels", STATUS_LABELS.get(status, status)))
        self.style().unpolish(self)
        self.style().polish(self)


def _pixmap_for_array(value: Any) -> QPixmap:
    import numpy as np

    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        pixels = np.asarray(array, dtype=np.float64)
        if pixels.size and np.nanmax(pixels) <= 1.0:
            pixels = pixels * 255.0
        pixels = np.nan_to_num(pixels, nan=0.0, posinf=255.0, neginf=0.0)
        pixels = np.ascontiguousarray(np.clip(pixels, 0, 255).astype(np.uint8))
        fmt = QImage.Format.Format_RGBA8888 if pixels.shape[-1] == 4 else QImage.Format.Format_RGB888
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], fmt).copy()
        return QPixmap.fromImage(image)
    if array.ndim != 2:
        raise ValueError(f"无法显示形状为 {array.shape} 的数组")
    numeric = np.asarray(array, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    if not finite.size:
        normalized = np.zeros(numeric.shape, dtype=np.float64)
    else:
        low, high = np.percentile(finite, (1, 99))
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        normalized = np.zeros(numeric.shape, dtype=np.float64) if high <= low else (numeric - low) / (high - low)
    normalized = np.clip(np.nan_to_num(normalized), 0.0, 1.0)
    # Compact perceptual blue -> cyan -> yellow map without a Matplotlib dependency.
    red = np.clip(2.2 * normalized - 0.35, 0, 1)
    green = np.clip(1.8 - np.abs(2.4 * normalized - 1.15), 0, 1)
    blue = np.clip(1.25 - 1.6 * normalized, 0, 1)
    pixels = np.ascontiguousarray(np.stack((red, green, blue), axis=-1) * 255, dtype=np.uint8)
    image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class ArrayImageWidget(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imagePanel")
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600;")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(110)
        self.image_label.setText(tr("ArrayImageWidget", "暂无数据"))
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)
        self._pixmap: QPixmap | None = None

    def set_array(self, value: Any | None, *, empty_text: str = "暂无数据") -> None:
        if value is None:
            self._pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(empty_text)
            return
        try:
            self._pixmap = _pixmap_for_array(value)
            self._update_pixmap()
        except (TypeError, ValueError) as exc:
            self._pixmap = None
            self.image_label.setText(str(exc))

    def set_image_path(self, path: str | Path | None) -> None:
        candidate = Path(path) if path else None
        if candidate is None or not candidate.is_file():
            self.set_array(None)
            return
        pixmap = QPixmap(str(candidate))
        if pixmap.isNull():
            self.set_array(None, empty_text=tr("ArrayImageWidget", "图片无法读取"))
            return
        self._pixmap = pixmap
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        size = self.image_label.size()
        self.image_label.setText("")
        self.image_label.setPixmap(
            self._pixmap.scaled(
                max(32, size.width()),
                max(32, size.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()


class LineChartWidget(QWidget):
    """Small dependency-free chart used for loss, MAE and histograms."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.series: dict[str, list[tuple[float, float]]] = {}
        self.setMinimumHeight(150)

    def set_series(self, series: Mapping[str, Any]) -> None:
        converted: dict[str, list[tuple[float, float]]] = {}
        for name, points in series.items():
            values: list[tuple[float, float]] = []
            for index, point in enumerate(points):
                if hasattr(point, "step") and hasattr(point, "value"):
                    values.append((float(point.step), float(point.value)))
                elif isinstance(point, (tuple, list)) and len(point) == 2:
                    values.append((float(point[0]), float(point[1])))
                elif isinstance(point, numbers.Real):
                    values.append((float(index), float(point)))
            if values:
                converted[str(name)] = values
        self.series = converted
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f8fafb"))
        bounds = self.rect().adjusted(42, 28, -14, -24)
        painter.setPen(QPen(QColor("#dce3e9"), 1))
        painter.drawRect(bounds)
        painter.setPen(QColor("#31465b"))
        painter.drawText(10, 18, self.title)
        if not self.series:
            painter.setPen(QColor("#8393a2"))
            painter.drawText(bounds, Qt.AlignmentFlag.AlignCenter, tr("LineChartWidget", "等待数据"))
            return
        all_points = [point for points in self.series.values() for point in points]
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        if max_x <= min_x:
            max_x = min_x + 1
        if max_y <= min_y:
            pad = abs(min_y) * 0.05 or 1.0
            min_y -= pad
            max_y += pad
        colors = (QColor("#176fb0"), QColor("#d47a1f"), QColor("#3b8c62"), QColor("#854fa0"))
        for color, (name, points) in zip(colors, self.series.items()):
            path = QPainterPath()
            for index, (x, y) in enumerate(points):
                px = bounds.left() + (x - min_x) / (max_x - min_x) * bounds.width()
                py = bounds.bottom() - (y - min_y) / (max_y - min_y) * bounds.height()
                path.moveTo(px, py) if index == 0 else path.lineTo(px, py)
            painter.setPen(QPen(color, 2))
            painter.drawPath(path)
            painter.drawText(bounds.right() - 105, 16 + 15 * list(self.series).index(name), name)


class GalleryCard(SectionCard):
    def __init__(self, sample: GallerySample, parent: QWidget | None = None) -> None:
        super().__init__(sample.title, sample.message, parent)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.input_panel = ArrayImageWidget(tr("GalleryCard", "输入概览"))
        self.target_panel = ArrayImageWidget(tr("GalleryCard", "目标深度"))
        self.prediction_panel = ArrayImageWidget(tr("GalleryCard", "预测深度"))
        self.error_panel = ArrayImageWidget(tr("GalleryCard", "绝对误差"))
        self.input_panel.set_array(sample.input_overview)
        self.prediction_panel.set_array(sample.prediction_depth)
        if sample.labeled:
            self.target_panel.set_array(sample.target_depth)
            self.error_panel.set_array(sample.absolute_error)
        else:
            self.target_panel.set_array(None, empty_text=tr("GalleryCard", "无标签，已隐藏"))
            self.error_panel.set_array(None, empty_text=tr("GalleryCard", "无标签，已隐藏"))
        grid.addWidget(self.input_panel, 0, 0)
        grid.addWidget(self.target_panel, 0, 1)
        grid.addWidget(self.prediction_panel, 0, 2)
        grid.addWidget(self.error_panel, 0, 3)
        self.body_layout.addLayout(grid)
        if all(value is None for value in (sample.input_overview, sample.target_depth, sample.prediction_depth)) and sample.composite_image:
            composite = ArrayImageWidget(tr("GalleryCard", "结果图"))
            composite.set_image_path(sample.composite_image)
            self.body_layout.addWidget(composite)


class SampleInspectorDialog(QDialog):
    def __init__(self, sample_path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("SampleInspectorDialog", "样本浏览"))
        self.resize(980, 720)
        initial = Path(sample_path).expanduser().resolve()
        siblings = sorted(initial.parent.glob("sample_*.npz")) or sorted(initial.parent.glob("*.npz"))
        self.sample_paths = siblings if initial in siblings else [initial]
        root = QVBoxLayout(self)
        navigation = QHBoxLayout()
        self.previous_button = QPushButton(tr("SampleInspectorDialog", "上一个"))
        self.sample_combo = QComboBox()
        self.sample_combo.setObjectName("sampleListCombo")
        self.sample_combo.addItems([path.name for path in self.sample_paths])
        self.next_button = QPushButton(tr("SampleInspectorDialog", "下一个"))
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.sample_combo, 1)
        navigation.addWidget(self.next_button)
        root.addLayout(navigation)
        self.title = QLabel()
        self.title.setObjectName("cardTitle")
        root.addWidget(self.title)
        self.tabs = QTabWidget()
        overview = QWidget()
        grid = QGridLayout(overview)
        self.rgb_panel = ArrayImageWidget(tr("SampleInspectorDialog", "RGB"))
        self.count_panel = ArrayImageWidget(tr("SampleInspectorDialog", "Count map"))
        self.depth_panel = ArrayImageWidget(tr("SampleInspectorDialog", "深度"))
        self.histogram = LineChartWidget(tr("SampleInspectorDialog", "中心像素时间直方图"))
        grid.addWidget(self.rgb_panel, 0, 0)
        grid.addWidget(self.count_panel, 0, 1)
        grid.addWidget(self.depth_panel, 1, 0)
        grid.addWidget(self.histogram, 1, 1)
        self.tabs.addTab(overview, tr("SampleInspectorDialog", "基础视图"))
        self.layer_page = QWidget()
        self.layer_grid = QGridLayout(self.layer_page)
        self.layer_tab_index = self.tabs.addTab(
            self.layer_page, tr("SampleInspectorDialog", "仿真专属图层")
        )
        self.fields = QLabel()
        self.fields.setWordWrap(True)
        self.fields.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        field_page = QWidget()
        field_layout = QVBoxLayout(field_page)
        field_layout.addWidget(self.fields)
        field_layout.addStretch(1)
        self.tabs.addTab(field_page, tr("SampleInspectorDialog", "字段"))
        root.addWidget(self.tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.previous_button.clicked.connect(lambda: self._step(-1))
        self.next_button.clicked.connect(lambda: self._step(1))
        self.sample_combo.currentIndexChanged.connect(self._load_current)
        initial_index = self.sample_paths.index(initial) if initial in self.sample_paths else 0
        self.sample_combo.setCurrentIndex(initial_index)
        self._load_current()

    def _step(self, offset: int) -> None:
        if not self.sample_paths:
            return
        index = (self.sample_combo.currentIndex() + offset) % len(self.sample_paths)
        self.sample_combo.setCurrentIndex(index)

    def _load_current(self, _index: int | None = None) -> None:
        index = self.sample_combo.currentIndex()
        if not 0 <= index < len(self.sample_paths):
            return
        inspection = load_sample_inspection(self.sample_paths[index])
        self.title.setText(
            f"{inspection.path.name} · {index + 1}/{len(self.sample_paths)} · {len(inspection.fields)} 个字段"
        )
        self.rgb_panel.set_array(
            inspection.rgb, empty_text=tr("SampleInspectorDialog", "样本不含 RGB")
        )
        self.count_panel.set_array(inspection.count_map)
        self.depth_panel.set_array(
            inspection.depth, empty_text=tr("SampleInspectorDialog", "无标签深度")
        )
        self.histogram.set_series(
            {"photon counts": inspection.histogram} if inspection.histogram is not None else {}
        )
        _clear_widget_layout(self.layer_grid)
        for layer_index, (name, value) in enumerate(inspection.layers.items()):
            panel = ArrayImageWidget(name)
            panel.set_array(value)
            self.layer_grid.addWidget(panel, layer_index // 2, layer_index % 2)
        self.tabs.setTabVisible(self.layer_tab_index, bool(inspection.layers))
        self.fields.setText("、".join(inspection.fields))
        self.previous_button.setEnabled(len(self.sample_paths) > 1)
        self.next_button.setEnabled(len(self.sample_paths) > 1)


def _clear_widget_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


__all__ = [
    "ArrayImageWidget",
    "DynamicParameterForm",
    "GalleryCard",
    "LineChartWidget",
    "PageHeader",
    "PathPicker",
    "SampleInspectorDialog",
    "SectionCard",
    "StatusBadge",
]
