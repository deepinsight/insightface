"""Model settings page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit, QTextEdit

from ..app import begin_context_activity, context_activity_count, end_context_activity
from ..core.config import save_config
from ..core.face_engine import FaceEngine, is_cuda_provider_available, providers_from_choice
from ..core.model_downloads import is_model_package_installed, list_installed_gfpgan_models, list_installed_swap_models
from ..core.model_packages import (
    CUSTOM_MODEL_CHOICE, GUI_MODEL_PACKAGES, PERSON_MODEL_PACKAGES,
    ensure_person_model, inspect_person_model, person_model_providers, person_provider_runtime_display,
)
from .base import BasePage


class ModelSettingsPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Model Settings", "Configure model packs, execution provider, face swap models, and runtime checks.", parent)
        form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_packages = list(GUI_MODEL_PACKAGES)
        self._rebuild_model_combo()
        custom_model_dir = (
            ""
            if context.config.model_name in self.model_packages
            else str(context.config.custom_model_dir or "").strip()
        )
        if not custom_model_dir and context.config.model_name not in self.model_packages:
            custom_model_dir = str(context.config.model_name or "").strip()
        self.custom_dir = QLineEdit(custom_model_dir)
        self.custom_dir_label = None
        self.model_root = QLineEdit(str(context.config.model_root))
        self.model_root.setPlaceholderText("~/.insightface")
        self.model_root.editingFinished.connect(self._update_model_availability)
        self._update_model_availability(sync_selection=True)
        self.model_combo.currentIndexChanged.connect(self._update_custom_dir_visibility)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Auto", "CPU", "CUDA"])
        self.provider_combo.setCurrentText(context.config.provider)
        self._update_provider_availability()
        self.det_combo = QComboBox()
        self.det_combo.addItems(["Auto", "128x128", "320x320", "640x640", "1024x1024"])
        self.det_combo.setCurrentText(context.config.det_size_label)
        self.model_combo.currentIndexChanged.connect(self._update_person_controls)
        self.swap_model_combo = QComboBox()
        self._update_swap_model_choices()
        self.gfpgan_enabled = QCheckBox("Enable GFPGAN restore after face swap")
        self.gfpgan_enabled.setChecked(bool(getattr(context.config, "enable_gfpgan", False)))
        self.gfpgan_model_combo = QComboBox()
        self._update_gfpgan_model_choices()
        form.addRow("Model package", self.model_combo)
        form.addRow("Model root", self.model_root)
        form.addRow("Custom model directory", self.custom_dir)
        self.custom_dir_label = form.labelForField(self.custom_dir)
        form.addRow("Provider", self.provider_combo)
        form.addRow("Detection size", self.det_combo)
        form.addRow("Face swap model", self.swap_model_combo)
        form.addRow("GFPGAN post-processing", self.gfpgan_enabled)
        form.addRow("GFPGAN model", self.gfpgan_model_combo)
        self.content.addLayout(form)
        self.runtime = QTextEdit()
        self.runtime.setReadOnly(True)
        self.content.addWidget(QLabel("Runtime information"))
        self.content.addWidget(self.runtime)
        self.content.addWidget(
            self.row(
                self.button("Save Settings", self.save),
                self.button("Open Model Downloads", lambda: self.window().open_page("Model Downloads")),
                self.button("Test Model Load", self.test_load),
                self.button("Warmup", self.warmup),
            )
        )
        self.refresh()
        self._update_custom_dir_visibility()
        self._update_person_controls()

    def _apply_to_config(self) -> None:
        cfg = self.context.config
        model_name, model_root, custom_model_dir = self._selected_model_values()
        provider = self.provider_combo.currentText()
        provider = (
            "Auto"
            if provider == "CUDA" and not is_cuda_provider_available()
            else provider
        )
        det_size = self._selected_det_size()
        cfg.model_name = model_name
        cfg.model_root = model_root
        cfg.custom_model_dir = custom_model_dir
        cfg.provider = provider
        cfg.det_size = [det_size[0], det_size[1]]
        cfg.swap_model_path = str(self.swap_model_combo.currentData() or "")
        cfg.gfpgan_model_path = str(self.gfpgan_model_combo.currentData() or "")
        cfg.enable_gfpgan = bool(
            self.gfpgan_enabled.isChecked() and cfg.gfpgan_model_path
        )

    def _selected_model_values(self) -> tuple[str, str, str]:
        chosen = self.model_combo.currentData()
        root = self.model_root.text().strip()
        if not root:
            raise ValueError("Model root must not be empty.")
        if chosen == CUSTOM_MODEL_CHOICE:
            custom_dir = self.custom_dir.text().strip()
            if not custom_dir:
                raise ValueError("Select a custom model directory.")
            # Keep the model identity non-empty for reports/embedding metadata,
            # while FaceEngine uses the explicit directory for resolution.
            return custom_dir, root, custom_dir
        return str(chosen), root, ""

    def _selected_det_size(self) -> tuple[int, int]:
        if self.det_combo.currentText() == "Auto":
            return (0, 0)
        width, height = self.det_combo.currentText().split("x")
        return int(width), int(height)

    def save(self) -> None:
        try:
            self._apply_to_config()
        except ValueError as exc:
            self.show_error(str(exc))
            return
        save_config(self.context.config)
        manager = self.window()
        if hasattr(manager, "notify_model_configuration_changed"):
            manager.notify_model_configuration_changed()
        self.set_status("Model settings saved.")
        self.refresh()

    def test_load(self) -> None:
        if any(context_activity_count(self.context, key) for key in (
            "person_analysis_jobs_in_progress", "privateframe_jobs_in_progress",
            "model_downloads_in_progress",
        )):
            self.show_error("Wait for active model or analysis work to finish before testing models.")
            return
        try:
            model_name, model_root, custom_model_dir = self._selected_model_values()
        except ValueError as exc:
            self.show_error(str(exc))
            return

        provider = self.provider_combo.currentText()
        if model_name in PERSON_MODEL_PACKAGES:
            self._test_person_load(model_name, model_root, provider)
            return
        provider = (
            "Auto"
            if provider == "CUDA" and not is_cuda_provider_available()
            else provider
        )
        det_size = self._selected_det_size()

        def task():
            engine = FaceEngine(
                model_name=model_name,
                providers=providers_from_choice(provider),
                det_size=det_size,
                root=model_root,
                custom_model_dir=custom_model_dir,
            )
            engine.load()
            return engine

        def done(engine):
            info = engine.get_runtime_info()
            self.runtime.setPlainText(
                "\n".join(f"{key}: {value}" for key, value in info.items())
            )
            if engine.is_loaded():
                self.set_status("Model loaded successfully.")
            else:
                self.show_error(engine.last_error or "Model load failed.")

        self.run_task("Loading model", task, done)

    def _test_person_load(self, model_name, model_root, provider) -> None:
        cache_dir = self.context.config.cache_dir

        def task(progress=None, is_cancelled=None):
            from ...app.person_analysis import PersonAnalysis

            providers = person_model_providers(provider)
            ensure_person_model(model_name, model_root, cache_dir, progress, is_cancelled)
            if is_cancelled is not None and is_cancelled():
                raise RuntimeError("Person model preparation cancelled.")
            app = PersonAnalysis(name=model_name, root=model_root, providers=providers)
            try:
                app.prepare()
                return {"model": model_name, "providers": providers}
            finally:
                app.close()

        def done(info):
            self.runtime.setPlainText("\n".join(f"{key}: {value}" for key, value in info.items()))
            self.set_status("Person Analysis models loaded, warmed up, and closed successfully.")

        begin_context_activity(self.context, "person_analysis_jobs_in_progress")

        def finished():
            end_context_activity(self.context, "person_analysis_jobs_in_progress")

        try:
            self.run_task("Loading Person Analysis models", task, done, on_finished=finished)
        except Exception:
            finished()
            raise

    def warmup(self) -> None:
        if self.model_combo.currentData() in PERSON_MODEL_PACKAGES:
            self.test_load()
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        self.run_task("Model warmup", self.context.engine.warmup, lambda info: self.set_status(f"Warmup complete: {info['warmup_ms']:.1f} ms"))

    def refresh(self) -> None:
        self._update_model_availability(sync_selection=True)
        self._update_provider_availability()
        self._update_swap_model_choices()
        self._update_gfpgan_model_choices()
        if self.context.config.model_name in PERSON_MODEL_PACKAGES:
            provider, detail = person_provider_runtime_display(self.context.config.provider)
            info = {"model": self.context.config.model_name, "provider": provider,
                    "provider_policy": detail, "status": "Prepared when Person Analysis starts."}
        else:
            info = self.context.engine.get_runtime_info()
        self.runtime.setPlainText("\n".join(f"{key}: {value}" for key, value in info.items()))

    def _update_provider_availability(self) -> None:
        cuda_available = is_cuda_provider_available()
        cuda_index = self.provider_combo.findText("CUDA")
        if cuda_index >= 0:
            item = self.provider_combo.model().item(cuda_index)
            if item is not None:
                item.setEnabled(cuda_available)
                item.setToolTip(
                    "CUDAExecutionProvider is available."
                    if cuda_available
                    else "CUDAExecutionProvider is not available. Install a matching onnxruntime-gpu, CUDA runtime, and GPU driver first."
                )
        if self.provider_combo.currentText() == "CUDA" and not cuda_available:
            self.provider_combo.setCurrentText("Auto")
            self.provider_combo.setToolTip(
                "CUDA is unavailable on this machine, so Auto will choose "
                "the best available provider."
            )
        elif self.model_combo.currentData() in PERSON_MODEL_PACKAGES:
            self.provider_combo.setToolTip(
                "Person Analysis Auto chooses CUDA when available, otherwise CPU."
            )
        else:
            self.provider_combo.setToolTip(
                "Auto chooses the best available ONNX Runtime provider in "
                "this order: CoreML, CUDA, CPU."
            )

    def _update_person_controls(self) -> None:
        is_person = self.model_combo.currentData() in PERSON_MODEL_PACKAGES
        self.det_combo.setEnabled(not is_person)
        self.det_combo.setToolTip(
            "Person Analysis uses the package input sizes: face 640; body 320 for cheetah_s or 640 for cheetah_l."
            if is_person else ""
        )
        self._update_provider_availability()

    def _update_model_availability(self, *, sync_selection: bool = False) -> None:
        model = self.model_combo.model()
        root = self.model_root.text().strip() or self.context.config.model_root
        for index, package in enumerate(self.model_packages):
            item = model.item(index)
            if item is None:
                continue
            installed = is_model_package_installed(package, root)
            person_status = inspect_person_model(package, root) if package in PERSON_MODEL_PACKAGES else None
            # Missing packages remain selectable: the general GUI reports that
            # a manual download is required, while PrivateFrame may download a
            # selected Raccoon package on first use.
            item.setEnabled(True)
            item.setText(package if installed else f"{package} (not downloaded)")
            item.setData(package, Qt.UserRole)
            if installed:
                item.setData(None, Qt.ForegroundRole)
            else:
                item.setForeground(QBrush(QColor("#9ca3af")))
            item.setToolTip(
                f"{package} is installed under {Path(root).expanduser() / 'models'}."
                if installed
                else f"{package} is not downloaded. Open Models > Downloads to install it."
            )
            if person_status is not None:
                item.setToolTip(person_status.message)
                if person_status.state == "invalid":
                    item.setText(f"{package} (invalid package)")
        custom_item = model.item(len(self.model_packages))
        if custom_item is not None:
            custom_item.setEnabled(True)
            custom_item.setText("custom model directory")
            custom_item.setData(CUSTOM_MODEL_CHOICE, Qt.UserRole)
        if sync_selection:
            current = (
                self.context.config.model_name
                if self.context.config.model_name in self.model_packages
                else CUSTOM_MODEL_CHOICE
            )
            index = self.model_combo.findData(current)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self._update_custom_dir_visibility()

    def _rebuild_model_combo(self) -> None:
        self.model_combo.clear()
        for package in self.model_packages:
            self.model_combo.addItem(package, package)
        self.model_combo.addItem("custom model directory", CUSTOM_MODEL_CHOICE)

    def _update_custom_dir_visibility(self) -> None:
        is_custom = self.model_combo.currentData() == CUSTOM_MODEL_CHOICE
        self.custom_dir.setVisible(is_custom)
        if hasattr(self, "custom_dir_label") and self.custom_dir_label is not None:
            self.custom_dir_label.setVisible(is_custom)

    def _update_swap_model_choices(self) -> None:
        current = getattr(self.context.config, "swap_model_path", "")
        paths = list_installed_swap_models(self.context.config.model_root)
        self.swap_model_combo.blockSignals(True)
        self.swap_model_combo.clear()
        if not paths:
            self.swap_model_combo.addItem("No downloaded swap models", "")
            self.swap_model_combo.setEnabled(False)
            self.swap_model_combo.setToolTip("Download inswapper_128.onnx from Models > Downloads first.")
        else:
            self.swap_model_combo.setEnabled(True)
            self.swap_model_combo.setToolTip("Only downloaded swap models are shown.")
            for path in paths:
                label = f"{Path(path).parent.name}/{Path(path).name}"
                self.swap_model_combo.addItem(label, str(path))
            index = self.swap_model_combo.findData(current)
            if index >= 0:
                self.swap_model_combo.setCurrentIndex(index)
            elif current and Path(current).exists():
                self.swap_model_combo.addItem(Path(current).name, current)
                self.swap_model_combo.setCurrentIndex(self.swap_model_combo.count() - 1)
        self.swap_model_combo.blockSignals(False)

    def _update_gfpgan_model_choices(self) -> None:
        current = getattr(self.context.config, "gfpgan_model_path", "")
        paths = list_installed_gfpgan_models(self.context.config.model_root)
        self.gfpgan_model_combo.blockSignals(True)
        self.gfpgan_model_combo.clear()
        if not paths:
            self.gfpgan_model_combo.addItem("No downloaded GFPGAN models", "")
            self.gfpgan_model_combo.setEnabled(False)
            self.gfpgan_enabled.setEnabled(False)
            self.gfpgan_model_combo.setToolTip("Download GFPGANv1.4.onnx from Models > Downloads first.")
            self.gfpgan_enabled.setToolTip("GFPGAN is unavailable until a GFPGAN model is downloaded.")
        else:
            self.gfpgan_model_combo.setEnabled(True)
            self.gfpgan_enabled.setEnabled(True)
            self.gfpgan_model_combo.setToolTip("Only downloaded GFPGAN restore models are shown.")
            self.gfpgan_enabled.setToolTip("Run GFPGAN face restoration after face swap.")
            for path in paths:
                label = f"{Path(path).parent.name}/{Path(path).name}"
                self.gfpgan_model_combo.addItem(label, str(path))
            index = self.gfpgan_model_combo.findData(current)
            if index >= 0:
                self.gfpgan_model_combo.setCurrentIndex(index)
            elif current and Path(current).exists():
                self.gfpgan_model_combo.addItem(Path(current).name, current)
                self.gfpgan_model_combo.setCurrentIndex(self.gfpgan_model_combo.count() - 1)
        self.gfpgan_model_combo.blockSignals(False)
