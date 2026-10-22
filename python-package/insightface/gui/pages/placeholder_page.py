"""Reusable placeholder and commercial next-steps pages."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton

from ..core.tooltips import set_button_tooltip
from .base import BasePage


class PlaceholderPage(BasePage):
    def __init__(
        self,
        context,
        title: str,
        description: str,
        coming_soon_text: str = "Coming soon.",
        ctas: list[tuple[str, str]] | None = None,
        parent=None,
    ):
        super().__init__(context, title, description, parent)
        status = QLabel(coming_soon_text)
        status.setWordWrap(True)
        status.setProperty("role", "muted")
        status.setStyleSheet("font-size: 16px; font-weight: 600; padding:12px 0;")
        self.content.addWidget(status)
        if ctas:
            buttons = []
            for label, page_key in ctas:
                button = QPushButton(label)
                button.clicked.connect(lambda checked=False, target=page_key: self.window().open_page(target))
                set_button_tooltip(button)
                buttons.append(button)
            self.content.addWidget(self.row(*buttons))
        self.content.addStretch(1)


class CommercialNextStepsPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Commercial Next Steps",
            "Understand the business paths after local enterprise evaluation.",
            parent,
        )
        self.content.addWidget(
            self.notice(
                "Production or commercial deployment requires an appropriate commercial model license. "
                "This page summarizes common commercial paths; it does not provide legal advice."
            )
        )
        summary = QLabel(
            "\n".join(
                [
                    "Commercial model license: authorize production use of approved model files.",
                    "Private model evaluation: validate models on your local business data before purchase.",
                    "SDK / API access: integrate InsightFace capabilities into internal products or services.",
                    "SLA and enterprise support: align support expectations for production deployment.",
                    "Custom training: evaluate private data requirements, consent, retention, and compliance before any training workflow.",
                ]
            )
        )
        summary.setWordWrap(True)
        summary.setStyleSheet("line-height: 1.35;")
        self.content.addWidget(summary)
        self.content.addWidget(
            self.row(
                self.button("Open License Center", lambda: self.window().open_license_dialog()),
                self.button("Run Evaluation", lambda: self.window().open_page("enterprise_evaluation")),
                self.button("Open Reports", lambda: self.window().open_page("reports")),
            )
        )
        self.content.addStretch(1)
