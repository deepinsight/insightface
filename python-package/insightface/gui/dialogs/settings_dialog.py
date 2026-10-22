"""Application settings dialog."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

from ..core.config import save_config
from ..core.i18n import LANGUAGE_OPTIONS, apply_translations, normalize_language, tr
from ..core.theme import THEME_OPTIONS, normalize_theme, theme_description


class SettingsDialog(QDialog):
    settingsSaved = Signal()

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("Settings")
        self.resize(620, 300)

        layout = QVBoxLayout(self)
        title = QLabel("Application Settings")
        title.setStyleSheet("font-size:18px; font-weight:700;")
        note = QLabel("Only appearance settings are configurable here. Workspace paths are fixed after first launch.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        layout.addWidget(title)
        layout.addWidget(note)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.theme = QComboBox()
        for option in THEME_OPTIONS:
            self.theme.addItem(option.label, option.value)
        current = normalize_theme(self.context.config.ui_theme)
        current_index = self.theme.findData(current)
        self.theme.setCurrentIndex(max(0, current_index))
        self.theme.currentIndexChanged.connect(self._update_theme_description)
        form.addRow("UI theme", self.theme)
        self.language = QComboBox()
        for option in LANGUAGE_OPTIONS:
            self.language.addItem(option.native_label, option.value)
        current_language = normalize_language(self.context.config.ui_language)
        language_index = self.language.findData(current_language)
        self.language.setCurrentIndex(max(0, language_index))
        self.language.currentIndexChanged.connect(self._update_theme_description)
        form.addRow("Language", self.language)
        layout.addLayout(form, 1)
        self.theme_help = QLabel("")
        self.theme_help.setWordWrap(True)
        self.theme_help.setProperty("role", "muted")
        layout.addWidget(self.theme_help)
        self._update_theme_description()

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).clicked.connect(self.save_and_close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        layout.addWidget(buttons)
        apply_translations(self, self.context.config.ui_language)

    def apply(self) -> None:
        self.context.config.ui_theme = self.theme.currentData()
        self.context.config.ui_language = self.language.currentData()
        save_config(self.context.config)
        self.settingsSaved.emit()
        apply_translations(self, self.context.config.ui_language)

    def save_and_close(self) -> None:
        self.apply()
        self.accept()

    def _update_theme_description(self) -> None:
        self.theme_help.setText(tr(theme_description(self.theme.currentData()), self.language.currentData()))
