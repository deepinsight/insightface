"""Application settings page."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QSpinBox

from ..core.config import AppConfig, save_config
from ..core.paths import ensure_workspace
from ..core.theme import THEME_OPTIONS, normalize_theme
from .base import BasePage


class SettingsPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Settings", "Configure workspace, database, export paths, privacy defaults, and UI preferences.", parent)
        cfg = context.config
        self.workspace = QLineEdit(cfg.workspace_path)
        self.database = QLineEdit(cfg.database_path)
        self.crops = QLineEdit(cfg.crop_dir)
        self.reports = QLineEdit(cfg.report_dir)
        self.exports = QLineEdit(cfg.export_dir)
        self.cache = QLineEdit(cfg.cache_dir)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(cfg.recognition_threshold)
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 100)
        self.top_k.setValue(cfg.default_top_k)
        self.min_det = QDoubleSpinBox()
        self.min_det.setRange(0, 1)
        self.min_det.setSingleStep(0.01)
        self.min_det.setValue(cfg.min_detection_score)
        self.min_face = QSpinBox()
        self.min_face.setRange(1, 4096)
        self.min_face.setValue(cfg.min_face_size)
        self.save_crops = QCheckBox("Save crops")
        self.save_crops.setChecked(cfg.save_crops)
        self.save_logs = QCheckBox("Save recognition logs")
        self.save_logs.setChecked(cfg.save_recognition_logs)
        self.anonymize = QCheckBox("Anonymize report paths")
        self.anonymize.setChecked(cfg.anonymize_report_paths)
        self.theme = QComboBox()
        for option in THEME_OPTIONS:
            self.theme.addItem(option.label, option.value)
        self.theme.setCurrentIndex(max(0, self.theme.findData(normalize_theme(cfg.ui_theme))))
        form = QFormLayout()
        for label, widget in [
            ("Workspace path", self.workspace),
            ("Database path", self.database),
            ("Crop output directory", self.crops),
            ("Report output directory", self.reports),
            ("Export output directory", self.exports),
            ("Cache directory", self.cache),
            ("Default recognition threshold", self.threshold),
            ("Default Top-K", self.top_k),
            ("Minimum detection score", self.min_det),
            ("Minimum face size", self.min_face),
            ("UI theme", self.theme),
        ]:
            form.addRow(label, widget)
        form.addRow("Privacy", self.row(self.save_crops, self.save_logs, self.anonymize))
        self.content.addLayout(form)
        self.content.addWidget(self.row(self.button("Save Settings", self.save), self.button("Open Workspace Folder", self.open_workspace), self.button("Export Settings", self.export_settings), self.button("Import Settings", self.import_settings), self.button("Reset Settings", self.reset)))

    def save(self) -> None:
        cfg = self.context.config
        cfg.workspace_path = self.workspace.text().strip()
        paths = ensure_workspace(cfg.workspace_path)
        cfg.database_path = self.database.text().strip() or str(paths["database"])
        cfg.crop_dir = self.crops.text().strip() or str(paths["crops"])
        cfg.report_dir = self.reports.text().strip() or str(paths["reports"])
        cfg.export_dir = self.exports.text().strip() or str(paths["exports"])
        cfg.cache_dir = self.cache.text().strip() or str(paths["cache"])
        cfg.recognition_threshold = float(self.threshold.value())
        cfg.default_top_k = int(self.top_k.value())
        cfg.min_detection_score = float(self.min_det.value())
        cfg.min_face_size = int(self.min_face.value())
        cfg.save_crops = self.save_crops.isChecked()
        cfg.save_recognition_logs = self.save_logs.isChecked()
        cfg.anonymize_report_paths = self.anonymize.isChecked()
        cfg.ui_theme = self.theme.currentData()
        save_config(cfg)
        self.set_status("Settings saved.")

    def open_workspace(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.workspace.text().strip()))

    def export_settings(self) -> None:
        path = self.save_file("Export settings", str(Path(self.context.config.export_dir) / "insightface_gui_settings.json"), "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(self.context.config.to_dict(), indent=2), encoding="utf-8")
        self.set_status(f"Settings exported to {path}")

    def import_settings(self) -> None:
        path = self.choose_file("Import settings", "JSON (*.json)")
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.context.config = AppConfig.from_dict(data)
        save_config(self.context.config)
        self.set_status("Settings imported. Restart recommended.")

    def reset(self) -> None:
        self.context.config = AppConfig()
        save_config(self.context.config)
        self.set_status("Settings reset to defaults. Restart recommended.")
