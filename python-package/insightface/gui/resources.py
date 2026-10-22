"""Application resources and desktop metadata helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from .core.constants import (
    APP_DOMAIN,
    APP_ID,
    APP_NAME,
    APP_ORGANIZATION,
    APP_PROCESS_NAME,
    APP_VERSION,
)

ASSET_DIR = Path(__file__).resolve().parent / "assets"
APP_ICON_SVG = ASSET_DIR / "app_icon.svg"
APP_ICON_PNG = ASSET_DIR / "app_icon.png"
APP_ICON_ICO = ASSET_DIR / "app_icon.ico"
APP_ICON_ICNS = ASSET_DIR / "app_icon.icns"


def app_icon_path() -> Path:
    """Return the best available runtime icon path."""

    for path in (APP_ICON_PNG, APP_ICON_ICO, APP_ICON_ICNS, APP_ICON_SVG):
        if path.exists():
            return path
    return APP_ICON_PNG


def app_icon():
    """Create a QIcon for the studio, returning an empty icon if Qt is unavailable."""

    try:
        from PySide6.QtGui import QIcon
    except Exception:
        return None
    for path in (APP_ICON_PNG, APP_ICON_ICO, APP_ICON_ICNS, APP_ICON_SVG):
        if not path.exists():
            continue
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon
    return QIcon()


def configure_application_metadata(app) -> None:
    """Apply stable, non-localized desktop metadata to the Qt application.

    Process and app identifiers intentionally remain English brand strings.
    User-facing UI text is localized separately; OS-level identifiers should stay
    stable for task managers, crash logs, permissions, packaging, and automation.
    """

    app.setApplicationName(APP_NAME)
    if hasattr(app, "setApplicationDisplayName"):
        app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORGANIZATION)
    app.setOrganizationDomain(APP_DOMAIN)
    if hasattr(app, "setDesktopFileName"):
        app.setDesktopFileName(APP_ID)
    icon = app_icon()
    if icon is not None and not icon.isNull():
        app.setWindowIcon(icon)
    _set_windows_app_user_model_id()
    _set_process_title_best_effort(APP_PROCESS_NAME)


def _set_windows_app_user_model_id() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def _set_process_title_best_effort(name: str) -> None:
    try:
        import setproctitle

        setproctitle.setproctitle(name)
    except Exception:
        pass


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "APP_ID",
    "APP_PROCESS_NAME",
    "APP_ICON_SVG",
    "APP_ICON_PNG",
    "APP_ICON_ICO",
    "APP_ICON_ICNS",
    "app_icon",
    "app_icon_path",
    "configure_application_metadata",
]
