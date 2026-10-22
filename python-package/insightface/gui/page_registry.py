"""Lazy page registry used by mode-based navigation."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from .pages.album_page import AlbumPage
from .pages.album_people_page import AlbumPeoplePage
from .pages.batch_processing_page import BatchProcessingPage
from .pages.camera_recognition_page import CameraRecognitionPage
from .pages.compare_page import ComparePage
from .pages.enterprise_eval_page import EnterpriseEvalPage
from .pages.face_search_page import FaceSearchPage
from .pages.face_swap_page import FaceSwapPage
from .pages.mode_dashboards import (
    AlbumDashboardPage,
    EnterpriseDashboardPage,
    FaceDashboardPage,
    SwapDashboardPage,
)
from .pages.multiface_photo_page import MultiFacePhotoPage
from .pages.people_library_page import PeopleLibraryPage
from .pages.person_analysis_page import PersonAnalysisPage
from .pages.placeholder_page import CommercialNextStepsPage, PlaceholderPage
from .pages.privateframe_page import PrivateFramePage
from .pages.reports_page import ReportsPage
from .pages.verification_page import VerificationPage
from .pages.video_search_page import VideoSearchPage


class PageRegistry:
    def __init__(self, context):
        self.context = context
        self._pages: dict[str, QWidget] = {}

    def get(self, page_key: str) -> QWidget:
        if page_key not in self._pages:
            self._pages[page_key] = self._create_page(page_key)
        return self._pages[page_key]

    @property
    def pages(self) -> dict[str, QWidget]:
        return self._pages

    def _create_page(self, page_key: str) -> QWidget:
        factories = {
            "person_analysis": PersonAnalysisPage,
            "private_frame": PrivateFramePage,
            "verification": VerificationPage,
            "face_dashboard": FaceDashboardPage,
            "album_dashboard": AlbumDashboardPage,
            "album": AlbumPage,
            "swap_dashboard": SwapDashboardPage,
            "enterprise_dashboard": EnterpriseDashboardPage,
            "compare": ComparePage,
            "face_search": FaceSearchPage,
            "multiface_photo": MultiFacePhotoPage,
            "people_library": PeopleLibraryPage,
            "batch_processing": BatchProcessingPage,
            "camera_recognition": CameraRecognitionPage,
            "video_search": VideoSearchPage,
            "album_import_scan": BatchProcessingPage,
            "album_people_clustering": AlbumPeoplePage,
            "album_person_search": FaceSearchPage,
            "image_face_swap": FaceSwapPage,
            "enterprise_evaluation": EnterpriseEvalPage,
            "reports": ReportsPage,
            "commercial_next_steps": CommercialNextStepsPage,
        }
        if page_key in factories:
            return factories[page_key](self.context)
        placeholders = {
            "album_multi_person_search": (
                "Multi-person Photo Search",
                "Find photos that contain multiple selected people.",
                "This page will search indexed album faces for photos containing combinations such as Alice and Bob.",
            ),
            "album_export": (
                "Export Album Results",
                "Export local album organization results.",
                "This page will export per-person folders, face crops, CSV summaries, and review queues.",
            ),
            "batch_image_swap": (
                "Batch Image Swap",
                "Swap faces across a folder of images using the configured swap model.",
                "Batch image swap is coming soon. Use Image Face Swap for single-image trials.",
            ),
            "video_face_swap": (
                "Video Face Swap",
                "Swap faces in local videos.",
                "Video Face Swap is coming soon and may require separate commercial authorization.",
            ),
            "source_face_library": (
                "Source Face Library",
                "Prepare high-quality source faces for future swap workflows.",
                "This page will manage reusable source faces, consent notes, and quality review.",
            ),
            "enterprise_dataset_setup": (
                "Dataset Setup",
                "Prepare local evaluation datasets and templates.",
                "Dataset setup is currently built into Run Evaluation. A guided setup workspace is coming soon.",
            ),
            "threshold_calibration": (
                "Threshold Calibration",
                "Review thresholds, FAR, FRR, and operating point recommendations.",
                "Threshold calibration tables are currently generated in evaluation reports. A dedicated calibration page is coming soon.",
            ),
        }
        title, description, body = placeholders.get(
            page_key,
            ("Coming Soon", "This workflow is planned for a future release.", "Coming soon."),
        )
        return PlaceholderPage(self.context, title, description, body)
