"""License dialog."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from ..core.i18n import tr
from ..pages.license_center_page import LicenseCenterPage


class LicenseDialog(QDialog):
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("License Center")
        self.resize(900, 660)
        layout = QVBoxLayout(self)
        self.page = LicenseCenterPage(context)
        layout.addWidget(self.page, 1)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def set_status(self, message: str) -> None:
        self.status_label.setText(tr(message, self.context.config.ui_language))
        if self.parent() is not None and hasattr(self.parent(), "set_status"):
            self.parent().set_status(message)
