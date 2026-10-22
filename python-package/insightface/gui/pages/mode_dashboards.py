"""Mode-specific dashboard pages."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QWidget

from ..core.constants import LOCAL_PROCESSING_NOTICE, RESPONSIBLE_USE_NOTICE
from ..core.tooltips import set_button_tooltip
from .base import BasePage


class ModeDashboardPage(BasePage):
    def __init__(
        self,
        context,
        title: str,
        description: str,
        cards: list[tuple[str, str, str, str]],
        notice: str = LOCAL_PROCESSING_NOTICE,
        parent=None,
    ):
        super().__init__(context, title, description, parent)
        self.content.addWidget(self.notice(notice))
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(12)
        for index, (card_title, body, page_key, action) in enumerate(cards):
            card = QWidget()
            card.setObjectName("dashboardCard")
            layout = QGridLayout(card)
            title_label = QLabel(card_title)
            title_label.setStyleSheet("font-size:16px; font-weight:700; border:0;")
            body_label = QLabel(body)
            body_label.setWordWrap(True)
            body_label.setProperty("role", "muted")
            button = QPushButton(action)
            button.clicked.connect(lambda checked=False, target=page_key: self.window().open_page(target))
            set_button_tooltip(button, body)
            layout.addWidget(title_label, 0, 0)
            layout.addWidget(body_label, 1, 0)
            layout.addWidget(button, 2, 0)
            grid.addWidget(card, index // 2, index % 2)
        self.content.addWidget(grid_widget)
        self.content.addStretch(1)


class FaceDashboardPage(ModeDashboardPage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Verification Dashboard",
            "Compare, identify, and recognize faces from images, folders, camera, and video.",
            [
                ("1:1 Compare", "Compare two local face images and export the decision.", "compare", "Open"),
                ("1:N Search", "Search a query face against your local People Library.", "face_search", "Open"),
                ("People Library", "Register people and manage local face samples.", "people_library", "Manage"),
                ("Batch Processing", "Scan folders and identify known or unknown faces.", "batch_processing", "Scan"),
                ("Camera Recognition", "Run live local recognition with consent.", "camera_recognition", "Open"),
                ("Video Search", "Find a registered person in a local video.", "video_search", "Open"),
            ],
            parent=parent,
        )


class AlbumDashboardPage(ModeDashboardPage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Album Dashboard",
            "Organize local photos by people, clusters, and searchable face groups.",
            [
                ("Scan Photo Library", "Index local image folders for face organization.", "album_import_scan", "Scan"),
                ("People Clustering", "Group unknown faces into approximate person clusters.", "album_people_clustering", "Cluster"),
                ("People Library", "Review known people and high-quality samples.", "people_library", "Manage"),
                ("Search by Person", "Find likely person matches in your local library.", "album_person_search", "Search"),
                ("Export Results", "Prepare local CSV and folder exports.", "album_export", "Review"),
            ],
            parent=parent,
        )


class SwapDashboardPage(ModeDashboardPage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Swap Dashboard",
            "Swap faces in images and videos using configured InsightFace swap models.",
            [
                ("Image Face Swap", "Try a local image swap with a configured swap model.", "image_face_swap", "Open"),
                ("Batch Image Swap", "Swap faces across a folder of images.", "batch_image_swap", "Preview"),
                ("Video Face Swap", "Video swap workflows are planned for a future version.", "video_face_swap", "Preview"),
                ("Source Face Library", "Prepare high-quality source faces for reuse.", "source_face_library", "Preview"),
            ],
            notice=(
                RESPONSIBLE_USE_NOTICE
                + " Face swap may require separate commercial authorization depending on usage and model license."
            ),
            parent=parent,
        )


class EnterpriseDashboardPage(ModeDashboardPage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Evaluation Dashboard",
            "Evaluate InsightFace models on local business datasets and generate procurement-ready reports.",
            [
                ("Run Evaluation", "Run KYC, 1:N, media, video, or swap evaluation scenarios.", "enterprise_evaluation", "Run"),
                ("Dataset Setup", "Prepare local datasets and CSV templates.", "enterprise_dataset_setup", "Review"),
                ("Reports", "Open and export historical evaluation reports.", "reports", "Open"),
                ("Threshold Calibration", "Review threshold operating points and recommendations.", "threshold_calibration", "Review"),
                ("Commercial Next Steps", "Understand licensing, SDK, API, SLA, and custom training paths.", "commercial_next_steps", "Open"),
            ],
            parent=parent,
        )
