"""Main-window composition and cross-page desktop workflows."""

from __future__ import annotations

from pathlib import Path

from spimaging.appcore.config import RunConfig
from spimaging.desktop.controller import WorkerController
from spimaging.desktop.dependency import require_pyside6
from spimaging.desktop.i18n import tr
from spimaging.desktop.models import (
    ApplicationPaths,
    DesktopSettings,
    PublicDemoAssets,
    ResultGalleryModel,
    RunHistoryModel,
    SettingsStore,
    clone_run_config,
    next_run_directory,
)
from spimaging.desktop.pages import (
    ExperimentPage,
    HistoryPage,
    HomePage,
    ResultsPage,
    RunPage,
    SettingsPage,
)


require_pyside6()

from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    PAGE_NAMES = ("home", "experiment", "run", "results", "history", "settings")

    def __init__(
        self,
        *,
        paths: ApplicationPaths | None = None,
        settings_store: SettingsStore | None = None,
        python_executable: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.paths = paths or ApplicationPaths.default()
        self.settings_store = settings_store or SettingsStore(self.paths.settings_file, self.paths)
        self.settings = self.settings_store.load()
        self.demo = PublicDemoAssets.discover()
        self.paths.ensure()
        self.history_model = RunHistoryModel(self.paths.history_db)
        self.history_model.recover_interrupted()
        self.worker = WorkerController(self, python_executable=python_executable)
        self._pending_close = False

        self.setWindowTitle(tr("MainWindow", "SPImaging 单光子成像工作台"))
        self.setMinimumSize(1120, 720)
        self.resize(1380, 900)
        self._build_ui()
        self._connect_workflows()
        self.refresh_history()
        self.navigate("home")

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(218)
        nav_layout = QVBoxLayout(sidebar)
        nav_layout.setContentsMargins(18, 24, 18, 20)
        nav_layout.setSpacing(7)
        brand = QLabel("SPImaging")
        brand.setObjectName("brandTitle")
        subtitle = QLabel(tr("MainWindow", "单光子成像实验工作台"))
        subtitle.setObjectName("brandSubtitle")
        nav_layout.addWidget(brand)
        nav_layout.addWidget(subtitle)
        nav_layout.addSpacing(24)

        labels = {
            "home": tr("MainWindow", "首页"),
            "experiment": tr("MainWindow", "新建实验"),
            "run": tr("MainWindow", "运行监控"),
            "results": tr("MainWindow", "结果画廊"),
            "history": tr("MainWindow", "历史与恢复"),
            "settings": tr("MainWindow", "设置"),
        }
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: dict[str, QPushButton] = {}
        for index, name in enumerate(self.PAGE_NAMES):
            button = QPushButton(labels[name])
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, target=name: self.navigate(target))
            self.nav_group.addButton(button, index)
            self.nav_buttons[name] = button
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)
        beta = QLabel("0.2.0-beta.1 · unsigned beta")
        beta.setObjectName("brandSubtitle")
        beta.setWordWrap(True)
        nav_layout.addWidget(beta)
        shell.addWidget(sidebar)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(54)
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(24, 0, 24, 0)
        self.context_label = QLabel()
        self.context_label.setStyleSheet("font-weight: 600; color: #42566b;")
        self.runtime_label = QLabel(tr("MainWindow", "本地运行 · 不上传数据"))
        self.runtime_label.setObjectName("muted")
        topbar_layout.addWidget(self.context_label)
        topbar_layout.addStretch(1)
        topbar_layout.addWidget(self.runtime_label)
        main_layout.addWidget(topbar)

        self.stack = QStackedWidget()
        self.home_page = HomePage(self.demo)
        self.experiment_page = ExperimentPage(self.paths, self.settings)
        self.run_page = RunPage()
        self.results_page = ResultsPage()
        self.history_page = HistoryPage(self.history_model, self.settings.runs_dir)
        self.settings_page = SettingsPage(self.settings_store, self.settings)
        self.pages = {
            "home": self.home_page,
            "experiment": self.experiment_page,
            "run": self.run_page,
            "results": self.results_page,
            "history": self.history_page,
            "settings": self.settings_page,
        }
        for name in self.PAGE_NAMES:
            self.stack.addWidget(self.pages[name])
        main_layout.addWidget(self.stack, 1)
        shell.addWidget(main, 1)
        self.setCentralWidget(central)
        self.run_page.bind_controller(self.worker)

    def _connect_workflows(self) -> None:
        self.home_page.quick_demo_requested.connect(self.start_quick_demo)
        self.home_page.new_experiment_requested.connect(lambda: self.navigate("experiment"))
        self.home_page.open_result_requested.connect(self._choose_result)
        self.home_page.history_requested.connect(lambda: self.navigate("history"))
        self.home_page.open_run_requested.connect(self.open_result)
        self.experiment_page.run_requested.connect(self.start_config)
        self.experiment_page.config_saved.connect(
            lambda path: self.statusBar().showMessage(tr("MainWindow", "配置已保存：") + path, 5000)
        )
        self.run_page.open_results_requested.connect(self.open_result)
        self.run_page.resume_requested.connect(self.prepare_resume)
        self.results_page.reuse_config_requested.connect(self.reuse_config)
        self.results_page.resume_requested.connect(self.prepare_resume)
        self.history_page.open_requested.connect(self.open_result)
        self.history_page.reuse_requested.connect(self._reuse_from_directory)
        self.history_page.resume_requested.connect(self.prepare_resume)
        self.settings_page.settings_saved.connect(self._settings_saved)
        self.settings_page.environment_repair_requested.connect(self._repair_environment)
        self.worker.completed.connect(lambda _status, _code: self.refresh_history())
        self.worker.completed.connect(self._finish_pending_close)

    def navigate(self, name: str) -> None:
        if name not in self.pages:
            raise ValueError(f"未知页面：{name}")
        self.stack.setCurrentWidget(self.pages[name])
        self.nav_buttons[name].setChecked(True)
        self.context_label.setText(self.nav_buttons[name].text())
        if name == "history":
            self.history_page.refresh()

    def refresh_history(self) -> None:
        records = self.history_model.list(limit=100)
        self.home_page.set_history(records)
        self.history_page.refresh()

    def start_quick_demo(self) -> None:
        if not self.demo.available:
            QMessageBox.warning(self, tr("MainWindow", "演示资产不可用"), self.demo.reason)
            return
        run_dir = next_run_directory(self.settings.runs_dir, "快速体验")
        config = RunConfig.new(
            "quick_demo",
            run_dir,
            display_name=tr("MainWindow", "合成样例快速体验"),
            input={
                "dataset_paths": [str(self.demo.dataset_dir)],
                "checkpoint_paths": [str(self.demo.checkpoint)],
            },
            visualization={"sample_count": 4},
            compute={"preference": self.settings.device, "gpu_index": self.settings.gpu_index},
            output={"history_db": str(self.paths.history_db)},
        )
        self.start_config(config)

    def start_config(self, config: RunConfig) -> None:
        if self.worker.is_running:
            QMessageBox.information(
                self,
                tr("MainWindow", "任务正在运行"),
                tr("MainWindow", "请等待当前任务完成或先安全取消。"),
            )
            self.navigate("run")
            return
        self.navigate("run")
        try:
            self.run_page.start_run(config)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(self, tr("MainWindow", "无法启动任务"), str(exc))

    def _choose_result(self) -> None:
        if self.results_page.choose_result():
            self.navigate("results")

    def open_result(self, run_dir: str) -> None:
        try:
            self.results_page.load_result(run_dir)
        except ValueError as exc:
            QMessageBox.warning(self, tr("MainWindow", "无法打开结果"), str(exc))
            return
        self.navigate("results")

    def reuse_config(self, config: RunConfig) -> None:
        self.experiment_page.load_config(config)
        self.navigate("experiment")
        self.statusBar().showMessage(tr("MainWindow", "已载入历史配置副本；运行前可继续修改。"), 5000)

    def _reuse_from_directory(self, run_dir: str) -> None:
        try:
            config = RunConfig.load(Path(run_dir) / "run.json")
        except ValueError as exc:
            QMessageBox.warning(self, tr("MainWindow", "无法复用配置"), str(exc))
            return
        self.reuse_config(config)

    def prepare_resume(self, run_dir: str) -> None:
        try:
            model = ResultGalleryModel.load(run_dir)
            checkpoint = model.compatible_resume_checkpoint()
            if checkpoint is None:
                raise ValueError("该任务没有兼容的恢复 checkpoint")
            clone = clone_run_config(
                model.config,
                next_run_directory(self.settings.runs_dir, model.config.display_name + "-resume"),
                resume_checkpoint=checkpoint,
            )
        except ValueError as exc:
            QMessageBox.warning(self, tr("MainWindow", "无法恢复任务"), str(exc))
            return
        self.experiment_page.load_config(clone)
        self.navigate("experiment")
        self.statusBar().showMessage(
            tr("MainWindow", "已载入兼容断点并增加目标 epoch；确认参数后开始运行。"), 7000
        )

    def _settings_saved(self, settings: DesktopSettings) -> None:
        self.settings = settings
        self.experiment_page.settings = settings
        self.experiment_page.output_picker.set_path(settings.runs_dir)
        self.history_page.runs_root = Path(settings.runs_dir)

    def _repair_environment(self) -> None:
        QMessageBox.information(
            self,
            tr("MainWindow", "运行环境修复"),
            tr(
                "MainWindow",
                "安装版会将修复请求交给 SPImaging 启动器，重新校验并原子切换私有运行时。源代码模式下请使用当前 Conda 环境修复依赖。",
            ),
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.worker.is_running:
            event.accept()
            return
        if self._pending_close:
            event.ignore()
            return
        answer = QMessageBox.question(
            self,
            tr("MainWindow", "任务仍在运行"),
            tr("MainWindow", "退出前需要安全取消当前任务。是否保存可恢复状态并退出？"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self._pending_close = True
        self.worker.request_cancel()
        self.statusBar().showMessage(tr("MainWindow", "正在安全取消；完成后将自动退出。"))
        event.ignore()

    def _finish_pending_close(self, _status: str, _code: int) -> None:
        if self._pending_close:
            QTimer.singleShot(0, self.close)


__all__ = ["MainWindow"]
