import os
import sys
from pathlib import Path

import pytest


def _platform_plugin_names():
    if sys.platform == "darwin":
        return {"libqcocoa.dylib", "libqoffscreen.dylib"}
    if sys.platform.startswith("win"):
        return {"qwindows.dll", "qoffscreen.dll"}
    if sys.platform.startswith("linux"):
        return {"libqxcb.so", "libqoffscreen.so"}
    return set()


def test_qt_platform_plugins_available_for_current_platform(monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.gui.app import configure_qt_plugin_paths, qt_plugin_root_candidates

    monkeypatch.delenv("QT_PLUGIN_PATH", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM_PLUGIN_PATH", raising=False)
    configure_qt_plugin_paths()

    assert os.environ.get("QT_PLUGIN_PATH"), (
        "QT_PLUGIN_PATH was not configured. "
        f"Qt plugin root candidates: {qt_plugin_root_candidates()}"
    )
    assert os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"), (
        "QT_QPA_PLATFORM_PLUGIN_PATH was not configured. "
        f"Qt plugin root candidates: {qt_plugin_root_candidates()}"
    )
    platform_dir = Path(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"])
    expected_plugins = _platform_plugin_names()
    if not expected_plugins:
        pytest.skip(f"No expected Qt platform plugin list for {sys.platform}")

    available_plugins = {path.name for path in platform_dir.iterdir()}
    missing_plugins = expected_plugins - available_plugins

    assert not missing_plugins, (
        f"Missing Qt platform plugins for {sys.platform}: "
        f"{sorted(missing_plugins)} in {platform_dir}"
    )


def test_qapplication_can_start_with_configured_offscreen_plugin(monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.gui.app import configure_qt_plugin_paths

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    configure_qt_plugin_paths()

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    assert app.platformName().lower() == "offscreen"
