"""Dashboard page."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QPushButton, QWidget

from ..core.constants import LOCAL_PROCESSING_NOTICE, SUBTITLE
from ..core.face_engine import provider_runtime_display
from ..core.i18n import tr
from ..core.licensing import current_model_license_display
from ..core.tooltips import set_button_tooltip
from ..widgets.metric_card import MetricCard
from .base import BasePage


class DashboardPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Dashboard", "Welcome to InsightFace Evaluation Studio.\n" + SUBTITLE, parent)
        provider_name, provider_tooltip = provider_runtime_display(
            context.config.provider
        )
        license_display = current_model_license_display(context)
        self.content.addWidget(self.notice(LOCAL_PROCESSING_NOTICE))
        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.cards = {
            "workspace": MetricCard("Workspace", context.config.workspace_path),
            "model": MetricCard("Model", context.config.model_name),
            "provider": MetricCard("Provider", provider_name),
            "license": MetricCard(
                "License",
                tr(license_display.status_text, context.config.ui_language),
            ),
            "people": MetricCard("People", "0"),
            "samples": MetricCard("Face samples", "0"),
            "media": MetricCard("Indexed photos", "0"),
            "faces": MetricCard("Detected faces", "0"),
        }
        for index, card in enumerate(self.cards.values()):
            self.grid.addWidget(card, index // 2, index % 2)
        self.cards["provider"].setToolTip(provider_tooltip)
        self.cards["license"].setToolTip(
            license_display.tooltip(context.config.ui_language)
        )
        self.content.addWidget(grid_widget)
        shortcuts = QWidget()
        shortcut_layout = QGridLayout(shortcuts)
        for index, (label, page) in enumerate(
            [
                ("Start 1:1 Compare", "1:1 Compare"),
                ("Add People", "People Library"),
                ("Scan Folder", "Batch Folder Processing"),
                ("Run Enterprise Evaluation", "Enterprise Evaluation"),
                ("Open License Center", "License Center"),
            ]
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, target=page: self.window().open_page(target))
            set_button_tooltip(button)
            shortcut_layout.addWidget(button, index // 3, index % 3)
        self.content.addWidget(shortcuts)
        self.content.addStretch(1)

    def refresh(self) -> None:
        counts = self.context.storage.counts()
        provider_name, provider_tooltip = provider_runtime_display(
            self.context.config.provider
        )
        license_display = current_model_license_display(self.context)
        self.cards["workspace"].set_value(self.context.config.workspace_path)
        self.cards["model"].set_value(self.context.config.model_name)
        self.cards["provider"].set_value(provider_name)
        self.cards["provider"].setToolTip(provider_tooltip)
        self.cards["license"].set_value(
            tr(license_display.status_text, self.context.config.ui_language)
        )
        self.cards["license"].setToolTip(
            license_display.tooltip(self.context.config.ui_language)
        )
        self.cards["people"].set_value(counts["people"])
        self.cards["samples"].set_value(counts["face_samples"])
        self.cards["media"].set_value(counts["media_items"])
        self.cards["faces"].set_value(counts["media_faces"])
