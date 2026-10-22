"""Mode-based navigation specification for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AppMode(str, Enum):
    PERSON_ANALYSIS = "person_analysis"
    PRIVATE_FRAME = "private_frame"
    FACE_VERIFICATION = "face_verification"
    ALBUM_MANAGEMENT = "album_management"
    FACE_SWAP = "face_swap"
    ENTERPRISE_EVALUATION = "enterprise_evaluation"


@dataclass(frozen=True)
class NavigationItem:
    id: str
    title: str
    page_key: str
    description: str = ""
    icon: str | None = None
    enabled: bool = True
    coming_soon: bool = False


@dataclass(frozen=True)
class NavigationMode:
    id: AppMode
    title: str
    description: str
    items: tuple[NavigationItem, ...]


NAVIGATION_MODES: dict[AppMode, NavigationMode] = {
    AppMode.PERSON_ANALYSIS: NavigationMode(
        id=AppMode.PERSON_ANALYSIS,
        title="Person Analysis",
        description="Compare people in a video or camera stream with registered references.",
        items=(
            NavigationItem(
                "person_analysis",
                "Person Analysis",
                "person_analysis",
                "Choose an input, add reference photos, and view the current matches.",
            ),
        ),
    ),
    AppMode.PRIVATE_FRAME: NavigationMode(
        id=AppMode.PRIVATE_FRAME,
        title="PrivateFrame",
        description="Redact faces in a local video and save the privacy-protected result.",
        items=(
            NavigationItem(
                "private_frame",
                "PrivateFrame Video Privacy",
                "private_frame",
                "Upload a video, choose privacy options, and create a redacted video locally.",
            ),
        ),
    ),
    AppMode.FACE_VERIFICATION: NavigationMode(
        id=AppMode.FACE_VERIFICATION,
        title="Face Recognition",
        description="Upload a query face and recognize it against one gallery image or a local gallery.",
        items=(
            NavigationItem("recognition", "Face Recognition", "verification", "Run 1:1 compare or 1:N gallery search from Query and Gallery uploads."),
        ),
    ),
    AppMode.ALBUM_MANAGEMENT: NavigationMode(
        id=AppMode.ALBUM_MANAGEMENT,
        title="Album Management",
        description="Organize local photos by people, clusters, and searchable face groups.",
        items=(
            NavigationItem("album", "Album", "album", "Import, refresh, cluster, and review local album folders."),
        ),
    ),
    AppMode.FACE_SWAP: NavigationMode(
        id=AppMode.FACE_SWAP,
        title="Face Swap",
        description="Use a configured downloaded swap model for local image or video face swap.",
        items=(
            NavigationItem("face_swap", "Face Swap", "image_face_swap", "Source + Target = Result for image or video swap."),
        ),
    ),
    AppMode.ENTERPRISE_EVALUATION: NavigationMode(
        id=AppMode.ENTERPRISE_EVALUATION,
        title="Enterprise Evaluation",
        description="Run local 1:1 and 1:N model evaluations and export reports.",
        items=(
            NavigationItem("enterprise_evaluation", "Enterprise Evaluation", "enterprise_evaluation", "Run local 1:1 or 1:N identity-folder evaluation and export reports."),
        ),
    ),
}


MODE_LABELS: dict[str, AppMode] = {mode.title: mode.id for mode in NAVIGATION_MODES.values()}


GLOBAL_PAGE_TITLES = {"Settings", "Model Settings", "Model Downloads", "License Center"}


def mode_from_value(value: str | AppMode | None) -> AppMode:
    if isinstance(value, AppMode):
        return value
    if value:
        for mode in AppMode:
            if value == mode.value:
                return mode
        for mode in NAVIGATION_MODES.values():
            if value == mode.title:
                return mode.id
    return AppMode.PERSON_ANALYSIS


def last_page_attr(mode: AppMode) -> str:
    return f"ui_last_page_{mode.value}"
