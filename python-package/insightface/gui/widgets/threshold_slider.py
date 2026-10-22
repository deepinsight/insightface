"""Threshold slider widget."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget
from PySide6.QtCore import Qt

from ..core.constants import DEFAULT_THRESHOLD


class ThresholdSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, value: float = DEFAULT_THRESHOLD, parent=None):
        super().__init__(parent)
        self.label = QLabel()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self._emit)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Threshold"))
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.set_value(value)

    def _emit(self, raw: int) -> None:
        value = raw / 100.0
        self.label.setText(f"{value:.2f}")
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self.slider.value() / 100.0

    def set_value(self, value: float) -> None:
        self.slider.setValue(int(round(float(value) * 100)))
        self.label.setText(f"{self.value():.2f}")
