"""Small metric display."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "0", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.title_label = QLabel(title)
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: object) -> None:
        self.value_label.setText(str(value))
