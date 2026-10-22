"""Cancelable progress dialog."""

from __future__ import annotations

from PySide6.QtWidgets import QProgressDialog
from PySide6.QtCore import Qt


class StudioProgressDialog(QProgressDialog):
    def __init__(self, title: str, parent=None):
        super().__init__("", "Cancel", 0, 100, parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumDuration(0)

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        total = max(1, int(total))
        self.setMaximum(total)
        self.setValue(min(int(current), total))
        self.setLabelText(message)
