"""Qt stylesheet helpers for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeOption:
    value: str
    label: str
    description: str


THEME_OPTIONS = [
    ThemeOption("system", "System", "Follow the operating system color scheme."),
    ThemeOption("precision_light", "Precision Light", "Neutral white and graphite for long review sessions."),
    ThemeOption("studio_dark", "Studio Dark", "Deep navy studio with violet and cyan model-workbench accents."),
    ThemeOption("graphite_pro", "Graphite Pro", "Charcoal enterprise console with amber command accents."),
    ThemeOption("azure_lab", "Azure Lab", "Bright blue laboratory workspace with crisp cyan interaction states."),
    ThemeOption("emerald_focus", "Emerald Focus", "Green privacy-first workspace for local biometric review."),
    ThemeOption("crimson_audit", "Crimson Audit", "Warm red audit theme for risk review and compliance workflows."),
]

_THEME_ALIASES = {
    "light": "precision_light",
    "dark": "studio_dark",
}

_LIGHT_THEME = "precision_light"
_DARK_THEME = "studio_dark"


def normalize_theme(theme: str | None) -> str:
    value = (theme or "system").strip().lower()
    value = _THEME_ALIASES.get(value, value)
    valid = {option.value for option in THEME_OPTIONS}
    return value if value in valid else "system"


def theme_label(theme: str | None) -> str:
    value = normalize_theme(theme)
    for option in THEME_OPTIONS:
        if option.value == value:
            return option.label
    return "System"


def theme_description(theme: str | None) -> str:
    value = normalize_theme(theme)
    for option in THEME_OPTIONS:
        if option.value == value:
            return option.description
    return THEME_OPTIONS[0].description


def effective_theme(theme: str | None) -> str:
    value = normalize_theme(theme)
    if value != "system":
        return value
    app = QApplication.instance()
    if app is not None:
        try:
            if app.styleHints().colorScheme() == Qt.ColorScheme.Dark:
                return _DARK_THEME
        except Exception:
            pass
    return _LIGHT_THEME


def _palettes() -> dict[str, dict[str, str]]:
    return {
        "precision_light": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#f5f5f2",
            "surface": "#ffffff",
            "surface_alt": "#eceee9",
            "panel": "#e2e4dd",
            "panel_hover": "#d9dcd4",
            "text": "#1f2428",
            "muted": "#636a70",
            "subtle": "#8a9096",
            "border": "#d0d3ca",
            "border_strong": "#a8ada4",
            "accent": "#2f3a45",
            "accent_2": "#5a6672",
            "accent_soft": "#dfe3e7",
            "accent_faint": "#f0f2f3",
            "field": "#ffffff",
            "button": "#ffffff",
            "button_hover": "#ebece8",
            "button_pressed": "#dfe3e7",
            "selection": "#d2d8dd",
            "success": "#2f855a",
            "success_soft": "#dcefe5",
            "warning": "#b7791f",
            "danger": "#c53030",
            "danger_hover": "#9b2c2c",
            "upload_bg": "#f7f7f4",
            "upload_hover": "#eceee9",
            "upload_drag": "#e2f2e8",
            "upload_file": "#ffffff",
            "notice_bg": "#eceee9",
            "notice_border": "#c8cdc4",
            "table_alt": "#f7f7f4",
            "header": "#e4e7e0",
            "chip": "#f4f5f2",
        },
        "studio_dark": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#070b1d",
            "surface": "#0d1533",
            "surface_alt": "#121d42",
            "panel": "#18295b",
            "panel_hover": "#213575",
            "text": "#edf3ff",
            "muted": "#a9b8d8",
            "subtle": "#7284ae",
            "border": "#27386f",
            "border_strong": "#4a63b3",
            "accent": "#7c5cff",
            "accent_2": "#24d6ff",
            "accent_soft": "#241d63",
            "accent_faint": "#111949",
            "field": "#080f28",
            "button": "#14204a",
            "button_hover": "#1b2d65",
            "button_pressed": "#251f6d",
            "selection": "#332b82",
            "success": "#2ee6a6",
            "success_soft": "#0e3a33",
            "warning": "#ffca4f",
            "danger": "#ff5c8a",
            "danger_hover": "#d93668",
            "upload_bg": "#0b1431",
            "upload_hover": "#13265a",
            "upload_drag": "#083840",
            "upload_file": "#0d1533",
            "notice_bg": "#121d54",
            "notice_border": "#3750a4",
            "table_alt": "#101a3d",
            "header": "#17265a",
            "chip": "#121d42",
        },
        "graphite_pro": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#13110d",
            "surface": "#1b1914",
            "surface_alt": "#252119",
            "panel": "#30291d",
            "panel_hover": "#3a3122",
            "text": "#f6f1e8",
            "muted": "#c2b8a8",
            "subtle": "#8d8375",
            "border": "#3e3527",
            "border_strong": "#6a5840",
            "accent": "#f59e0b",
            "accent_2": "#facc15",
            "accent_soft": "#4a320d",
            "accent_faint": "#2a2114",
            "field": "#12100d",
            "button": "#272218",
            "button_hover": "#3a3122",
            "button_pressed": "#4a320d",
            "selection": "#62440f",
            "success": "#84cc16",
            "success_soft": "#26350f",
            "warning": "#f97316",
            "danger": "#f43f5e",
            "danger_hover": "#be123c",
            "upload_bg": "#19160f",
            "upload_hover": "#2f2515",
            "upload_drag": "#27340f",
            "upload_file": "#1b1914",
            "notice_bg": "#2a2114",
            "notice_border": "#76541c",
            "table_alt": "#211d16",
            "header": "#30291d",
            "chip": "#272218",
        },
        "azure_lab": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#e5f2ff",
            "surface": "#fafdff",
            "surface_alt": "#d8ebff",
            "panel": "#c3ddff",
            "panel_hover": "#aed0ff",
            "text": "#09264a",
            "muted": "#31577d",
            "subtle": "#6685a4",
            "border": "#a9cbf3",
            "border_strong": "#5b9bed",
            "accent": "#075eea",
            "accent_2": "#00a4ff",
            "accent_soft": "#bfe0ff",
            "accent_faint": "#eef8ff",
            "field": "#ffffff",
            "button": "#fafdff",
            "button_hover": "#d8ebff",
            "button_pressed": "#bfe0ff",
            "selection": "#93c5fd",
            "success": "#009b72",
            "success_soft": "#c9f5e7",
            "warning": "#c27200",
            "danger": "#d92d5b",
            "danger_hover": "#a61d42",
            "upload_bg": "#fafdff",
            "upload_hover": "#d8ebff",
            "upload_drag": "#c9f5ff",
            "upload_file": "#ffffff",
            "notice_bg": "#d8ebff",
            "notice_border": "#77b5f7",
            "table_alt": "#eef7ff",
            "header": "#cfe4ff",
            "chip": "#e8f4ff",
        },
        "emerald_focus": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#e8f7ef",
            "surface": "#f8fff9",
            "surface_alt": "#d7f2e2",
            "panel": "#bfe8d1",
            "panel_hover": "#a8ddc2",
            "text": "#0d2f22",
            "muted": "#3b6b58",
            "subtle": "#6f927f",
            "border": "#9bd2b5",
            "border_strong": "#4fa878",
            "accent": "#07844f",
            "accent_2": "#00b894",
            "accent_soft": "#b9ead1",
            "accent_faint": "#eefbf3",
            "field": "#ffffff",
            "button": "#f8fff9",
            "button_hover": "#d7f2e2",
            "button_pressed": "#b9ead1",
            "selection": "#8bdcb2",
            "success": "#079455",
            "success_soft": "#c6f6d5",
            "warning": "#b7791f",
            "danger": "#c2415b",
            "danger_hover": "#9f1239",
            "upload_bg": "#f8fff9",
            "upload_hover": "#d7f2e2",
            "upload_drag": "#c6f6d5",
            "upload_file": "#ffffff",
            "notice_bg": "#d7f2e2",
            "notice_border": "#63bd8e",
            "table_alt": "#eefaf3",
            "header": "#cdebdc",
            "chip": "#e9f8ef",
        },
        "crimson_audit": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#190b10",
            "surface": "#241018",
            "surface_alt": "#321522",
            "panel": "#451b2c",
            "panel_hover": "#5a2438",
            "text": "#fff1f4",
            "muted": "#e5a9b5",
            "subtle": "#ae7280",
            "border": "#653046",
            "border_strong": "#9f4965",
            "accent": "#ff3b6b",
            "accent_2": "#ff8a3d",
            "accent_soft": "#5a1830",
            "accent_faint": "#32101f",
            "field": "#1b0c12",
            "button": "#351724",
            "button_hover": "#512236",
            "button_pressed": "#641d36",
            "selection": "#7f2946",
            "success": "#22c55e",
            "success_soft": "#123422",
            "warning": "#f97316",
            "danger": "#ff3b6b",
            "danger_hover": "#be123c",
            "upload_bg": "#1f0d15",
            "upload_hover": "#42182b",
            "upload_drag": "#2d2a10",
            "upload_file": "#241018",
            "notice_bg": "#3a1427",
            "notice_border": "#8c3a55",
            "table_alt": "#2b1320",
            "header": "#3b1828",
            "chip": "#321522",
        },
    }


def application_stylesheet(theme: str | None) -> str:
    palette = _palettes()[effective_theme(theme)]

    return f"""
    * {{
        font-family: {palette["font"]};
        font-size: 13px;
    }}
    QMainWindow, QDialog {{
        background: {palette["bg"]};
        color: {palette["text"]};
    }}
    QWidget {{
        color: {palette["text"]};
        selection-background-color: {palette["selection"]};
        selection-color: {palette["text"]};
    }}
    QWidget#topAppBar {{
        background: {palette["surface"]};
        border-bottom: 1px solid {palette["border"]};
    }}
    QWidget#modeRail {{
        background: {palette["surface"]};
        border-right: 1px solid {palette["border"]};
    }}
    QWidget#modeSidebar {{
        background: {palette["surface_alt"]};
        border-right: 1px solid {palette["border"]};
    }}
    QLabel[role="muted"], QLabel[role="status"], QLabel[role="secondary"] {{
        color: {palette["muted"]};
    }}
    QLabel[role="statusChip"] {{
        padding: 4px 9px;
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        background: {palette["chip"]};
        color: {palette["text"]};
        font-weight: 600;
    }}
    QLabel#noticeLabel {{
        background: {palette["notice_bg"]};
        border: 1px solid {palette["notice_border"]};
        border-radius: 6px;
        color: {palette["text"]};
        padding: 9px 10px;
    }}
    QFrame#privateFrameCompatibilityBanner {{
        background: {palette["notice_bg"]};
        border: 1px solid {palette["notice_border"]};
        border-radius: 8px;
    }}
    QLabel#privateFrameCompatibilityTitle {{
        color: {palette["text"]};
        font-weight: 700;
    }}
    QLabel#privateFrameCompatibilityMessage,
    QLabel#privateFrameCompatibilityCurrentModelCaption {{
        color: {palette["text"]};
    }}
    QLabel#privateFrameCompatibilityCurrentModelValue {{
        color: {palette["text"]};
        font-weight: 650;
    }}
    QLabel#localProcessingNotice {{
        background: {palette["notice_bg"]};
        border: 1px solid {palette["notice_border"]};
        border-radius: 8px;
        color: {palette["muted"]};
        padding: 9px 10px;
        line-height: 1.25;
    }}
    QFrame#downloadSourceFooter {{
        background: {palette["surface_alt"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
    }}
    QPushButton {{
        background: {palette["button"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        padding: 7px 12px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {palette["button_hover"]};
        border-color: {palette["border_strong"]};
    }}
    QPushButton:pressed {{
        background: {palette["button_pressed"]};
        border-color: {palette["accent"]};
    }}
    QPushButton:disabled {{
        color: {palette["subtle"]};
        background: {palette["surface_alt"]};
        border-color: {palette["border"]};
    }}
    QPushButton#primaryButton {{
        background: {palette["accent"]};
        border-color: {palette["accent"]};
        color: #ffffff;
    }}
    QPushButton#primaryButton:hover {{
        background: {palette["accent_2"]};
        border-color: {palette["accent_2"]};
    }}
    QPushButton#removeUpload {{
        background: {palette["button_pressed"]};
        color: {palette["text"]};
        border: 1px solid {palette["border_strong"]};
        border-radius: 11px;
        font-weight: 700;
        padding: 0;
    }}
    QPushButton#removeUpload:hover {{
        background: {palette["danger"]};
        border-color: {palette["danger"]};
        color: #ffffff;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
        background: {palette["field"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        padding: 6px 8px;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {palette["accent"]};
        background: {palette["surface"]};
    }}
    QComboBox::drop-down {{
        border: 0;
        width: 24px;
    }}
    QAbstractItemView {{
        background: {palette["surface"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        selection-background-color: {palette["selection"]};
    }}
    QComboBox QAbstractItemView::item:disabled {{
        color: {palette["subtle"]};
        background: {palette["surface_alt"]};
    }}
    QListWidget {{
        background: transparent;
        border: 0;
        outline: 0;
    }}
    QListWidget::item {{
        border-radius: 6px;
        padding: 8px 10px;
        margin: 2px 0;
    }}
    QListWidget::item:selected {{
        background: {palette["accent_soft"]};
        color: {palette["text"]};
    }}
    QListWidget::item:hover {{
        background: {palette["button_hover"]};
    }}
    QListWidget#modeList::item {{
        border-radius: 8px;
        padding: 10px 10px;
        margin: 3px 0;
        color: {palette["muted"]};
        border: 1px solid transparent;
    }}
    QListWidget#modeList::item:selected {{
        background: {palette["accent_soft"]};
        color: {palette["text"]};
        border: 1px solid {palette["accent"]};
    }}
    QListWidget#modeList::item:hover {{
        background: {palette["button_hover"]};
        color: {palette["text"]};
    }}
    QTableWidget, QTableView {{
        background: {palette["surface"]};
        alternate-background-color: {palette["table_alt"]};
        color: {palette["text"]};
        gridline-color: {palette["border"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        selection-background-color: {palette["selection"]};
    }}
    QTableWidget::item, QTableView::item {{
        padding: 4px;
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background: {palette["selection"]};
        color: {palette["text"]};
    }}
    QHeaderView::section {{
        background: {palette["header"]};
        color: {palette["text"]};
        border: 0;
        border-bottom: 1px solid {palette["border"]};
        padding: 7px;
        font-weight: 650;
    }}
    QFrame, QWidget#dashboardCard {{
        border-color: {palette["border"]};
    }}
    QWidget#dashboardCard {{
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        background: {palette["surface"]};
    }}
    QWidget#dashboardCard:hover {{
        border-color: {palette["accent"]};
        background: {palette["surface_alt"]};
    }}
    QFrame#enterpriseCard {{
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        background: {palette["surface"]};
        padding: 10px;
    }}
    QLabel#enterpriseCardTitle {{
        color: {palette["text"]};
    }}
    QFrame#uploadPreview, QFrame#privateFrameVideoInput, QFrame#dropInput, QFrame#galleryUpload,
    QFrame#imageOrFolderImport, QListWidget#albumDirectoryList {{
        border: 1px dashed {palette["border_strong"]};
        border-radius: 6px;
        background: {palette["upload_bg"]};
        padding: 8px;
    }}
    QFrame#uploadPreview[hoverActive="true"], QFrame#privateFrameVideoInput[hoverActive="true"],
    QFrame#dropInput[hoverActive="true"],
    QFrame#galleryUpload[hoverActive="true"], QFrame#imageOrFolderImport[hoverActive="true"],
    QListWidget#albumDirectoryList[hoverActive="true"] {{
        border-color: {palette["accent"]};
        background: {palette["upload_hover"]};
    }}
    QFrame#uploadPreview[dragActive="true"], QFrame#privateFrameVideoInput[dragActive="true"],
    QFrame#dropInput[dragActive="true"],
    QFrame#galleryUpload[dragActive="true"], QFrame#imageOrFolderImport[dragActive="true"],
    QListWidget#albumDirectoryList[dragActive="true"] {{
        border-color: {palette["success"]};
        background: {palette["upload_drag"]};
    }}
    QFrame#uploadPreview[hasFile="true"], QFrame#privateFrameVideoInput[hasFile="true"],
    QFrame#dropInput[hasFiles="true"],
    QFrame#galleryUpload[hasFiles="true"] {{
        border: 1px solid {palette["border"]};
        background: {palette["upload_file"]};
    }}
    QFrame#uploadPreview QLabel, QFrame#privateFrameVideoInput QLabel,
    QFrame#dropInput QLabel, QFrame#galleryUpload QLabel,
    QFrame#imageOrFolderImport QLabel {{
        background: transparent;
    }}
    QFrame#uploadPreview QGraphicsView, QFrame#privateFrameVideoInput QGraphicsView,
    QFrame#galleryUpload QListWidget {{
        background: transparent;
        border: 0;
    }}
    QGraphicsView#imageViewer {{
        background: {palette["upload_bg"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
    }}
    QWidget#imageViewerViewport {{
        background: {palette["upload_bg"]};
    }}
    QLabel#uploadPrompt, QLabel#dropPrompt {{
        color: {palette["muted"]};
        font-size: 15px;
        font-weight: 650;
        padding: 18px;
    }}
    QLabel#pathLabel {{
        color: {palette["muted"]};
        padding: 0 8px 8px 8px;
    }}
    QMenuBar, QMenu {{
        background: {palette["surface"]};
        color: {palette["text"]};
        border-color: {palette["border"]};
    }}
    QMenuBar::item:selected, QMenu::item:selected {{
        background: {palette["accent_soft"]};
    }}
    QTabWidget::pane {{
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        background: {palette["surface"]};
    }}
    QTabBar::tab {{
        background: {palette["surface_alt"]};
        border: 1px solid {palette["border"]};
        border-bottom: 0;
        padding: 7px 12px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }}
    QTabBar::tab:selected {{
        background: {palette["surface"]};
        color: {palette["accent"]};
    }}
    QProgressBar {{
        background: {palette["surface_alt"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        color: {palette["text"]};
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {palette["accent"]};
        border-radius: 5px;
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: {palette["surface_alt"]};
        border: 1px solid {palette["border"]};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {palette["accent"]};
        border: 1px solid {palette["accent"]};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSplitter::handle {{
        background: {palette["bg"]};
    }}
    QScrollArea {{
        background: transparent;
        border: 0;
    }}
    QStatusBar {{
        background: {palette["surface"]};
        color: {palette["muted"]};
        border-top: 1px solid {palette["border"]};
    }}
    QToolTip {{
        background: {palette["surface"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        border-radius: 4px;
        padding: 4px;
    }}
    """
