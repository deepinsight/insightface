"""Camera recognition page."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QSpinBox

from ..core.camera import list_camera_indices
from ..core.constants import RESPONSIBLE_USE_NOTICE
from ..widgets.threshold_slider import ThresholdSlider
from .base import BasePage


class CameraRecognitionPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Camera Recognition", "Live recognition preview for lawful, consent-based controlled environments.", parent)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 16)
        self.frame_skip = QSpinBox()
        self.frame_skip.setRange(1, 60)
        self.frame_skip.setValue(context.config.camera_frame_skip)
        self.threshold = ThresholdSlider(context.config.recognition_threshold)
        self.content.addWidget(self.row(QLabel("Camera index"), self.camera_index, QLabel("Frame skip"), self.frame_skip, self.button("Check Cameras", self.check), self.button("Start", self.unavailable), self.button("Stop", self.unavailable), self.button("Register Current Face", self.unavailable)))
        self.content.addWidget(self.threshold)
        self.content.addWidget(self.notice(RESPONSIBLE_USE_NOTICE + " Live recognition should only be used with consent and in lawful, controlled environments."))
        self.content.addStretch(1)

    def check(self) -> None:
        indices = list_camera_indices()
        self.set_status("Available camera indices: " + (", ".join(map(str, indices)) if indices else "none detected"))

    def unavailable(self) -> None:
        self.show_error("Live camera loop is not enabled in this first desktop build. Camera settings and responsible use notice are available.")
