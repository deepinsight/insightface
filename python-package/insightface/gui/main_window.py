"""Main window, mode navigation, and task orchestration."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFontMetrics, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .core.config import save_config
from .core.constants import APP_VERSION, LOCAL_PROCESSING_NOTICE, RESPONSIBLE_USE_NOTICE, WINDOW_TITLE
from .core.face_engine import is_cuda_provider_available, provider_runtime_display
from .core.i18n import apply_translations, tr
from .core.links import open_insightface_url
from .core.licensing import (
    current_model_license_display,
    invalidate_model_license_cache,
)
from .core.navigation import (
    AppMode,
    NAVIGATION_MODES,
    last_page_attr,
    mode_from_value,
)
from .core.model_packages import (
    CUSTOM_MODEL_CHOICE, GUI_MODEL_PACKAGES, PERSON_MODEL_PACKAGES,
    person_provider_runtime_display,
)
from .core.theme import application_stylesheet
from .core.tooltips import apply_button_tooltips, set_button_tooltip
from .dialogs.license_dialog import LicenseDialog
from .dialogs.model_manager_dialog import ModelManagerDialog
from .dialogs.settings_dialog import SettingsDialog
from .page_registry import PageRegistry
from .pages.license_center_page import ENTERPRISE_HELP_URL
from .resources import app_icon
from .widgets.progress_dialog import StudioProgressDialog


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            signature = inspect.signature(self.fn)
            if "progress" in signature.parameters or "is_cancelled" in signature.parameters:
                result = self.fn(
                    progress=lambda current, total, message="": self.signals.progress.emit(int(current), int(total), str(message)),
                    is_cancelled=lambda: self.cancelled,
                )
            else:
                result = self.fn()
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class FirstLaunchWizard(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("First Launch Wizard")
        self.config = config
        layout = QVBoxLayout(self)
        intro = QLabel(
            "\n".join(
                [
                    "Welcome to InsightFace Evaluation Studio.",
                    LOCAL_PROCESSING_NOTICE,
                    RESPONSIBLE_USE_NOTICE,
                ]
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.workspace = QLineEdit(config.workspace_path)
        browse = QLabel("<a href='#'>Browse</a>")
        browse.linkActivated.connect(self.browse_workspace)
        workspace_row = QWidget()
        workspace_layout = QVBoxLayout(workspace_row)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.addWidget(self.workspace)
        workspace_layout.addWidget(browse)
        self.mode = QComboBox()
        self.mode.setProperty("i18nItems", True)
        self.mode.addItem("Personal / Research", "Personal / Research")
        self.mode.addItem("Enterprise Evaluation", "Enterprise Evaluation")
        self.model = QComboBox()
        for package in GUI_MODEL_PACKAGES:
            self.model.addItem(package, package)
        self.model.addItem("custom model directory", CUSTOM_MODEL_CHOICE)
        configured_model = (
            config.model_name
            if config.model_name in GUI_MODEL_PACKAGES
            else CUSTOM_MODEL_CHOICE
        )
        configured_index = self.model.findData(configured_model)
        if configured_index >= 0:
            self.model.setCurrentIndex(configured_index)
        custom_value = (
            ""
            if config.model_name in GUI_MODEL_PACKAGES
            else str(config.custom_model_dir or "").strip()
        )
        if not custom_value and config.model_name not in GUI_MODEL_PACKAGES:
            custom_value = str(config.model_name or "").strip()
        self.custom_model_dir = QLineEdit(custom_value)
        self.custom_model_dir.setPlaceholderText("Choose a local model directory")
        self.model.currentIndexChanged.connect(self._update_custom_model_visibility)
        self.provider = QComboBox()
        self.provider.addItems(["Auto", "CPU", "CUDA"])
        self.provider.setCurrentText(str(config.provider))
        self._update_provider_availability()
        self.license_notice = QLabel(
            "This application runs locally. Code and model licenses may differ. "
            "Research models may be limited to non-commercial use. Commercial deployment "
            "requires appropriate model license. This tool does not provide legal advice."
        )
        self.license_notice.setWordWrap(True)
        form.addRow("Workspace", workspace_row)
        form.addRow("Mode", self.mode)
        form.addRow("Model package", self.model)
        form.addRow("Custom model directory", self.custom_model_dir)
        self.custom_model_dir_label = form.labelForField(self.custom_model_dir)
        form.addRow("Provider", self.provider)
        form.addRow("License notice", self.license_notice)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_custom_model_visibility()
        apply_button_tooltips(self)

    def browse_workspace(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select workspace", str(Path.home()))
        if folder:
            self.workspace.setText(folder)

    def accept(self) -> None:
        self.config.workspace_path = self.workspace.text().strip()
        self.config.mode = str(self.mode.currentData() or self.mode.currentText())
        self.config.ui_last_mode = (
            AppMode.ENTERPRISE_EVALUATION.value
            if self.config.mode == "Enterprise Evaluation"
            else AppMode.PERSON_ANALYSIS.value
        )
        selected_model = self.model.currentData()
        if selected_model == CUSTOM_MODEL_CHOICE:
            custom_dir = self.custom_model_dir.text().strip()
            if not custom_dir:
                QMessageBox.warning(
                    self,
                    "First Launch Wizard",
                    "Select a custom model directory.",
                )
                return
            self.config.model_name = custom_dir
            self.config.custom_model_dir = custom_dir
        else:
            self.config.model_name = str(selected_model)
            self.config.custom_model_dir = ""
        provider = self.provider.currentText()
        self.config.provider = "Auto" if provider == "CUDA" and not is_cuda_provider_available() else provider
        self.config.apply_workspace_defaults()
        save_config(self.config)
        super().accept()

    def _update_custom_model_visibility(self, *_args) -> None:
        visible = self.model.currentData() == CUSTOM_MODEL_CHOICE
        self.custom_model_dir.setVisible(visible)
        self.custom_model_dir_label.setVisible(visible)

    def _update_provider_availability(self) -> None:
        cuda_available = is_cuda_provider_available()
        cuda_index = self.provider.findText("CUDA")
        if cuda_index >= 0:
            item = self.provider.model().item(cuda_index)
            if item is not None:
                item.setEnabled(cuda_available)
                item.setToolTip(
                    "CUDAExecutionProvider is available."
                    if cuda_available
                    else "CUDAExecutionProvider is not available. Install a matching onnxruntime-gpu, CUDA runtime, and GPU driver first."
                )
        self.provider.setToolTip(
            "Auto chooses the best available ONNX Runtime provider in this "
            "order: CoreML, CUDA, CPU."
        )


class MainWindow(QMainWindow):
    LEGACY_PAGE_MAP = {
        "Person Analysis": "person_analysis",
        "PrivateFrame": "private_frame",
        "PrivateFrame Video Privacy": "private_frame",
        "Dashboard": "face_dashboard",
        "Face Recognition": "verification",
        "Verification": "verification",
        "Verification Dashboard": "face_dashboard",
        "Album Dashboard": "album_dashboard",
        "Swap Dashboard": "swap_dashboard",
        "Evaluation Dashboard": "enterprise_dashboard",
        "1:1 Compare": "compare",
        "1:N Face Search": "face_search",
        "Multi-face Photo Recognition": "multiface_photo",
        "Batch Folder Processing": "batch_processing",
        "Camera Recognition": "camera_recognition",
        "Video Person Search": "video_search",
        "People Library": "people_library",
        "Album People Clustering": "album_people_clustering",
        "Enterprise Evaluation": "enterprise_evaluation",
        "Reports": "reports",
        "Image Face Swap": "image_face_swap",
    }

    GLOBAL_PAGES = {"Settings", "Model Settings", "Model Downloads", "License Center"}

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.apply_theme()
        self.setWindowTitle(WINDOW_TITLE)
        icon = app_icon()
        if icon is not None and not icon.isNull():
            self.setWindowIcon(icon)
        self.resize(int(context.config.ui_window_width), int(context.config.ui_window_height))
        self.thread_pool = QThreadPool.globalInstance()
        self.active_workers: set[Worker] = set()
        self._model_autoload_running = False
        self._model_autoload_pending = False
        self._closing_after_workers = False
        self._initializing = True
        self.page_registry = PageRegistry(context)
        self.pages = self.page_registry.pages
        self.current_mode = mode_from_value(context.config.ui_last_mode)
        self.current_page_key = ""

        self.stack = QStackedWidget()
        self.mode_rail = QWidget()
        self.mode_rail.setObjectName("modeRail")
        self.mode_rail.setFixedWidth(230)
        self.mode_list = QListWidget()
        self.mode_list.setObjectName("modeList")
        self.mode_list.setAlternatingRowColors(False)
        self.mode_list.setWordWrap(True)
        self.mode_list.setSpacing(2)
        self.mode_list.itemClicked.connect(self._mode_list_clicked)
        self.sidebar = QWidget()
        self.sidebar.setObjectName("modeSidebar")
        self.sidebar.setFixedWidth(int(context.config.ui_sidebar_width or 240))
        self.sidebar_title = QLabel()
        self.sidebar_title.setStyleSheet("font-size:18px; font-weight:700;")
        self.sidebar_description = QLabel()
        self.sidebar_description.setWordWrap(True)
        self.sidebar_description.setProperty("role", "muted")
        self.sidebar_list = QListWidget()
        self.sidebar_list.setAlternatingRowColors(False)
        self.sidebar_list.itemClicked.connect(self._sidebar_clicked)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_top_bar())
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self._build_mode_rail())
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(14, 14, 10, 14)
        sidebar_layout.addWidget(self.sidebar_title)
        sidebar_layout.addWidget(self.sidebar_description)
        sidebar_layout.addWidget(self.sidebar_list, 1)
        body_layout.addWidget(self.sidebar)
        body_layout.addWidget(self.stack, 1)
        root.addWidget(body, 1)
        self.setCentralWidget(central)

        self._build_menu()
        self._build_statusbar()
        apply_button_tooltips(self)
        self.change_mode(self.current_mode, restore_last=True, save=False)
        self._initializing = False
        self.apply_language()
        self.refresh_statusbar()
        if not self.context.config_exists:
            QTimer.singleShot(100, self.show_first_launch)
        QTimer.singleShot(250, self._auto_load_model_if_needed)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topAppBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        title = QLabel("InsightFace Evaluation Studio")
        title.setStyleSheet("font-size:18px; font-weight:700; border:0;")
        version = QLabel(f"v{APP_VERSION}")
        version.setProperty("role", "muted")
        self.model_chip = QLabel()
        self.provider_chip = QLabel()
        self.license_chip = QLabel()
        for chip in (self.model_chip, self.provider_chip, self.license_chip):
            chip.setProperty("role", "statusChip")
        layout.addWidget(title)
        layout.addWidget(version)
        layout.addStretch(1)
        layout.addWidget(self.model_chip)
        layout.addWidget(self.provider_chip)
        layout.addWidget(self.license_chip)
        layout.addWidget(self._top_button("Models", self.open_model_manager))
        layout.addWidget(self._top_button("License", self.open_license_dialog))
        layout.addWidget(self._top_button("Settings", self.open_settings_dialog))
        return bar

    def _top_button(self, text: str, callback: Callable) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        set_button_tooltip(button)
        return button

    def _build_mode_rail(self) -> QWidget:
        layout = QVBoxLayout(self.mode_rail)
        layout.setContentsMargins(14, 14, 12, 14)
        layout.setSpacing(8)
        title = QLabel("Workflows")
        title.setObjectName("modeRailTitle")
        title.setStyleSheet("font-size:14px; font-weight:700; border:0;")
        description = QLabel("Choose the workspace for the task.")
        description.setProperty("role", "muted")
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)
        for mode in NAVIGATION_MODES.values():
            item = QListWidgetItem(f"{mode.title}\n{self._mode_subtitle(mode.id)}")
            item.setData(Qt.UserRole, mode.id.value)
            item.setToolTip(mode.description)
            item.setSizeHint(QSize(190, 58))
            self.mode_list.addItem(item)
        layout.addWidget(self.mode_list, 1)
        local_notice = QLabel("All processing is local. No images, embeddings, or reports are uploaded automatically.")
        local_notice.setObjectName("localProcessingNotice")
        local_notice.setWordWrap(True)
        layout.addWidget(local_notice)
        layout.addSpacing(6)
        enterprise_help = QPushButton("Enterprise Help")
        enterprise_help.setObjectName("enterpriseHelpButton")
        enterprise_help.setToolTip(
            "Open the InsightFace contact page for enterprise evaluation, licensing, SDK/API, SLA, or custom training help."
        )
        enterprise_help.clicked.connect(self.open_enterprise_help)
        layout.addWidget(enterprise_help)
        return self.mode_rail

    def _mode_subtitle(self, mode: AppMode) -> str:
        subtitles = {
            AppMode.PERSON_ANALYSIS: "Detect and match people",
            AppMode.PRIVATE_FRAME: "Local video face redaction",
            AppMode.FACE_VERIFICATION: "Query and gallery recognition",
            AppMode.ALBUM_MANAGEMENT: "Photo clustering and review",
            AppMode.FACE_SWAP: "Source + Target = Result",
            AppMode.ENTERPRISE_EVALUATION: "Datasets, metrics, and reports",
        }
        return tr(subtitles[mode], self.context.config.ui_language)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_workspace = file_menu.addAction("Open Workspace Folder")
        open_workspace.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.context.config.workspace_path)))
        quit_action = file_menu.addAction("Exit")
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)

        mode_menu = self.menuBar().addMenu("Mode")
        for mode in NAVIGATION_MODES:
            action = mode_menu.addAction(NAVIGATION_MODES[mode].title)
            action.triggered.connect(lambda checked=False, target=mode: self.change_mode(target))

        tools_menu = self.menuBar().addMenu("Tools")
        models_action = QAction("Models", self)
        models_action.setShortcut(QKeySequence("Ctrl+M"))
        models_action.triggered.connect(self.open_model_manager)
        settings_action = QAction("Settings", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self.open_settings_dialog)
        license_action = QAction("License Center", self)
        license_action.setShortcut(QKeySequence("Ctrl+L"))
        license_action.triggered.connect(self.open_license_dialog)
        tools_menu.addAction(models_action)
        tools_menu.addAction(settings_action)
        tools_menu.addAction(license_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(
            lambda: QMessageBox.information(
                self,
                tr("About", self.context.config.ui_language),
                f"InsightFace Evaluation Studio v{APP_VERSION}",
            )
        )
        responsible_action = help_menu.addAction("Responsible Use Notice")
        responsible_action.triggered.connect(
            lambda: QMessageBox.information(
                self,
                tr("Responsible Use Notice", self.context.config.ui_language),
                tr(RESPONSIBLE_USE_NOTICE, self.context.config.ui_language),
            )
        )

    def _build_statusbar(self) -> None:
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_labels = {
            "model": QLabel(),
            "provider": QLabel(),
            "database": QLabel(),
            "license": QLabel(),
            "running": QLabel("Ready"),
        }
        for label in self.status_labels.values():
            self.status.addPermanentWidget(label)

    def _mode_list_clicked(self, item: QListWidgetItem) -> None:
        mode = mode_from_value(item.data(Qt.UserRole))
        if mode != self.current_mode:
            self.change_mode(mode)

    def change_mode(self, mode: AppMode | str, restore_last: bool = True, save: bool = True) -> None:
        self.current_mode = mode_from_value(mode)
        self._sync_mode_list_selection()
        self._rebuild_sidebar()
        page_key = getattr(self.context.config, last_page_attr(self.current_mode), "") if restore_last else ""
        valid_keys = {item.page_key for item in NAVIGATION_MODES[self.current_mode].items}
        if page_key not in valid_keys:
            page_key = NAVIGATION_MODES[self.current_mode].items[0].page_key
        if save:
            self.context.config.ui_last_mode = self.current_mode.value
            save_config(self.context.config)
        self.open_page(page_key, from_mode_change=True)
        if not self._initializing:
            QTimer.singleShot(0, self._auto_load_model_if_needed)

    def _sync_mode_list_selection(self) -> None:
        for row in range(self.mode_list.count()):
            item = self.mode_list.item(row)
            if item.data(Qt.UserRole) == self.current_mode.value:
                self.mode_list.blockSignals(True)
                self.mode_list.setCurrentItem(item)
                self.mode_list.blockSignals(False)
                break

    def _rebuild_sidebar(self) -> None:
        mode = NAVIGATION_MODES[self.current_mode]
        self.sidebar_title.setText(tr(mode.title, self.context.config.ui_language))
        self.sidebar_description.setText(tr(mode.description, self.context.config.ui_language))
        self.sidebar.setVisible(False)
        self.sidebar_list.clear()
        for nav_item in mode.items:
            title = tr(nav_item.title, self.context.config.ui_language) + (
                "  " + tr("Coming soon", self.context.config.ui_language) if nav_item.coming_soon else ""
            )
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, nav_item.page_key)
            item.setToolTip(tr(nav_item.description, self.context.config.ui_language))
            if not nav_item.enabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setForeground(Qt.gray)
            self.sidebar_list.addItem(item)

    def _sidebar_clicked(self, item: QListWidgetItem) -> None:
        if item.flags() & Qt.ItemIsEnabled:
            self.open_page(str(item.data(Qt.UserRole)))

    def open_page(self, page_key: str, from_mode_change: bool = False) -> None:
        if page_key in self.GLOBAL_PAGES:
            if page_key == "Settings":
                self.open_settings_dialog()
            elif page_key in {"Model Settings", "Model Downloads"}:
                self.open_model_manager(initial=page_key)
            elif page_key == "License Center":
                self.open_license_dialog()
            return
        page_key = self.LEGACY_PAGE_MAP.get(page_key, page_key)
        page = self.page_registry.get(page_key)
        if self.stack.indexOf(page) < 0:
            self.stack.addWidget(page)
        apply_button_tooltips(page)
        if hasattr(page, "refresh"):
            page.refresh()
        apply_translations(page, self.context.config.ui_language)
        self.stack.setCurrentWidget(page)
        self.current_page_key = page_key
        for row in range(self.sidebar_list.count()):
            item = self.sidebar_list.item(row)
            if item.data(Qt.UserRole) == page_key:
                self.sidebar_list.setCurrentItem(item)
                break
        if not from_mode_change:
            setattr(self.context.config, last_page_attr(self.current_mode), page_key)
            save_config(self.context.config)
        self.set_status(self._page_title(page_key))

    def _page_title(self, page_key: str) -> str:
        for mode in NAVIGATION_MODES.values():
            for item in mode.items:
                if item.page_key == page_key:
                    return tr(item.title, self.context.config.ui_language)
        return page_key

    def show_first_launch(self) -> None:
        wizard = FirstLaunchWizard(self.context.config, self)
        apply_translations(wizard, self.context.config.ui_language)
        if wizard.exec() == QDialog.Accepted:
            self.context.storage = self.context.storage.__class__(self.context.config.database_path)
            self._model_configuration_changed()
            self.change_mode(mode_from_value(self.context.config.ui_last_mode), restore_last=False)
            self.refresh_statusbar()

    def _auto_load_model_if_needed(self) -> None:
        if (
            self._closing_after_workers
            or self.current_mode in {AppMode.PRIVATE_FRAME, AppMode.PERSON_ANALYSIS}
            or self.context.config.model_name in PERSON_MODEL_PACKAGES
            or not self.context.config.auto_load_model
            or self.context.runtime_safe_mode
            or self.context.engine.is_loaded()
            or self._model_autoload_running
        ):
            return
        if self.active_workers:
            QTimer.singleShot(250, self._auto_load_model_if_needed)
            return
        self.auto_load_model()

    def auto_load_model(self) -> None:
        if (
            self.current_mode in {AppMode.PRIVATE_FRAME, AppMode.PERSON_ANALYSIS}
            or self.context.config.model_name in PERSON_MODEL_PACKAGES
            or self._model_autoload_running
            or self.context.engine.is_loaded()
        ):
            return
        self._model_autoload_running = True
        engine = self.context.engine

        def task():
            engine.load()
            return engine

        def done(engine):
            self.refresh_statusbar()
            if engine is not self.context.engine:
                return
            if engine.is_loaded():
                self.set_status("Model loaded.")
            elif engine.last_error:
                self.set_status(engine.last_error)

        self.run_task(
            "Loading model",
            task,
            done,
            show_dialog=False,
            on_finished=self._model_autoload_finished,
        )

    def _model_autoload_finished(self) -> None:
        self._model_autoload_running = False
        if self._model_autoload_pending:
            self._model_autoload_pending = False
            QTimer.singleShot(0, self._auto_load_model_if_needed)

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.context, self)
        apply_button_tooltips(dialog)
        apply_translations(dialog, self.context.config.ui_language)
        dialog.settingsSaved.connect(self._settings_saved)
        dialog.exec()
        self.refresh_statusbar()

    def _settings_saved(self) -> None:
        self.apply_theme()
        self.apply_language()
        self.refresh_statusbar()

    def apply_theme(self) -> None:
        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.setStyleSheet(application_stylesheet(self.context.config.ui_theme))

    def apply_language(self) -> None:
        self._refresh_mode_list_texts()
        self._rebuild_sidebar()
        apply_translations(self, self.context.config.ui_language)
        self.refresh_statusbar()

    def _refresh_mode_list_texts(self) -> None:
        for row, mode in enumerate(NAVIGATION_MODES.values()):
            item = self.mode_list.item(row)
            if item is None:
                continue
            item.setText(f"{tr(mode.title, self.context.config.ui_language)}\n{self._mode_subtitle(mode.id)}")
            item.setToolTip(tr(mode.description, self.context.config.ui_language))

    def open_model_manager(self, initial: str | None = None) -> None:
        from .app import context_activity_count

        message = None
        if context_activity_count(self.context, "person_analysis_jobs_in_progress"):
            message = "Stop Person Analysis before opening Models."
        elif context_activity_count(self.context, "privateframe_jobs_in_progress"):
            message = "Wait for PrivateFrame processing to finish before opening Models."
        if message:
            self.set_status(message)
            QMessageBox.warning(
                self,
                tr("Models", self.context.config.ui_language),
                tr(message, self.context.config.ui_language),
            )
            return
        dialog = ModelManagerDialog(self.context, self)
        apply_button_tooltips(dialog)
        apply_translations(dialog, self.context.config.ui_language)
        dialog.modelChanged.connect(self._model_configuration_changed)
        dialog.modelFilesChanged.connect(self._model_files_changed)
        if initial:
            dialog.open_page(initial)
            apply_translations(dialog, self.context.config.ui_language)
        dialog.exec()
        self._model_files_changed()

    def _model_configuration_changed(self) -> None:
        """Propagate one global model selection to the engine and open pages."""

        from .app import reconfigure_context_engine

        engine_changed = reconfigure_context_engine(self.context)
        self._model_files_changed()
        if (
            engine_changed
            and self.current_mode not in {AppMode.PRIVATE_FRAME, AppMode.PERSON_ANALYSIS}
            and self.context.config.model_name not in PERSON_MODEL_PACKAGES
        ):
            if self._model_autoload_running:
                self._model_autoload_pending = True
            else:
                QTimer.singleShot(0, self._auto_load_model_if_needed)

    def _model_files_changed(self) -> None:
        """Refresh model readiness without invalidating a configured engine."""

        invalidate_model_license_cache()
        for page in tuple(self.pages.values()):
            if hasattr(page, "refresh"):
                page.refresh()
        self.refresh_statusbar()

    def open_license_dialog(self) -> None:
        dialog = LicenseDialog(self.context, self)
        apply_button_tooltips(dialog)
        apply_translations(dialog, self.context.config.ui_language)
        dialog.exec()
        self.refresh_statusbar()

    def open_enterprise_help(self) -> None:
        open_insightface_url(ENTERPRISE_HELP_URL, content="sidebar_enterprise_help")
        self.set_status("Opened InsightFace enterprise contact page.")

    def set_status(self, message: str) -> None:
        self.status_labels["running"].setText(self._elide(tr(message, self.context.config.ui_language), 180))

    def refresh_statusbar(self) -> None:
        cfg = self.context.config
        provider_name, provider_tooltip = (
            person_provider_runtime_display(cfg.provider, cfg.ui_language)
            if cfg.model_name in PERSON_MODEL_PACKAGES
            else provider_runtime_display(cfg.provider)
        )
        license_display = current_model_license_display(self.context)
        license_status = tr(license_display.status_text, cfg.ui_language)
        license_tooltip = license_display.tooltip(cfg.ui_language)
        self.model_chip.setVisible(cfg.ui_show_status_chips)
        self.provider_chip.setVisible(cfg.ui_show_status_chips)
        self.license_chip.setVisible(cfg.ui_show_status_chips)
        self.model_chip.setText(self._elide(f"{tr('Model', cfg.ui_language)}: {cfg.model_name}", 150))
        self.provider_chip.setText(
            self._elide(
                f"{tr('Provider', cfg.ui_language)}: {provider_name}",
                230,
            )
        )
        self.license_chip.setText(self._elide(license_status, 180))
        self.model_chip.setToolTip(cfg.model_name)
        self.provider_chip.setToolTip(provider_tooltip)
        self.license_chip.setToolTip(license_tooltip)
        self.status_labels["model"].setText(self._elide(f"{tr('Model', cfg.ui_language)}: {cfg.model_name}", 160))
        self.status_labels["provider"].setText(
            f"{tr('Provider', cfg.ui_language)}: {provider_name}"
        )
        self.status_labels["provider"].setToolTip(provider_tooltip)
        self.status_labels["database"].setText(self._elide(f"{tr('DB', cfg.ui_language)}: {cfg.database_path}", 280))
        self.status_labels["database"].setToolTip(cfg.database_path)
        self.status_labels["license"].setText(
            self._elide(
                f"{tr('License', cfg.ui_language)}: {license_status}",
                220,
            )
        )
        self.status_labels["license"].setToolTip(license_tooltip)

    def run_task(
        self,
        title: str,
        fn: Callable,
        on_result: Callable | None = None,
        show_dialog: bool = True,
        on_progress: Callable | None = None,
        on_error: Callable | None = None,
        on_finished: Callable | None = None,
    ) -> Worker:
        worker = Worker(fn)
        worker.setAutoDelete(False)
        self.active_workers.add(worker)
        dialog = StudioProgressDialog(title, self) if show_dialog else None
        if dialog:
            dialog.canceled.connect(worker.cancel)
            worker.signals.progress.connect(dialog.update_progress)
            dialog.show()
        if on_progress:
            worker.signals.progress.connect(on_progress)
        worker.signals.result.connect(lambda result: on_result(result) if on_result else None)
        if on_error:
            worker.signals.error.connect(on_error)
        else:
            worker.signals.error.connect(
                lambda message: QMessageBox.warning(
                    self,
                    tr(title, self.context.config.ui_language),
                    tr(message, self.context.config.ui_language),
                )
            )
            worker.signals.error.connect(self.set_status)
        worker.signals.finished.connect(lambda: dialog.close() if dialog else None)
        if on_finished:
            worker.signals.finished.connect(on_finished)
        worker.signals.finished.connect(self.refresh_statusbar)
        worker.signals.finished.connect(lambda: self.active_workers.discard(worker))
        worker.signals.finished.connect(worker.signals.deleteLater)
        self.set_status(title)
        self.thread_pool.start(worker)
        return worker

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.active_workers:
            if not self._closing_after_workers:
                answer = QMessageBox.question(
                    self,
                    tr("Processing Is Still Running", self.context.config.ui_language),
                    tr(
                        "Cancel the running task and close the application?",
                        self.context.config.ui_language,
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    event.ignore()
                    return
                self._closing_after_workers = True
                self.setEnabled(False)
                self.set_status("Cancelling running task before exit…")
                for worker in tuple(self.active_workers):
                    worker.cancel()
                QTimer.singleShot(50, self._close_when_workers_finish)
            event.ignore()
            return
        self._closing_after_workers = True
        self.context.config.ui_window_width = max(800, self.width())
        self.context.config.ui_window_height = max(600, self.height())
        self.context.config.ui_sidebar_width = max(200, self.sidebar.width())
        save_config(self.context.config)
        super().closeEvent(event)

    def _close_when_workers_finish(self) -> None:
        if self.active_workers:
            QTimer.singleShot(50, self._close_when_workers_finish)
            return
        self.close()

    def _elide(self, text: str, width: int) -> str:
        metrics = QFontMetrics(self.font())
        return metrics.elidedText(str(text), Qt.ElideMiddle, width)
