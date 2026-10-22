import os
import json

import pytest

from insightface.gui.core.config import AppConfig, load_config, save_config
from insightface.gui.core.navigation import AppMode, GLOBAL_PAGE_TITLES, NAVIGATION_MODES, mode_from_value


def test_navigation_modes_do_not_include_global_sidebar_items():
    assert set(NAVIGATION_MODES) == set(AppMode)
    all_page_keys = []
    for mode in NAVIGATION_MODES.values():
        assert mode.items
        assert len(mode.items) <= 8
        for item in mode.items:
            assert item.title not in GLOBAL_PAGE_TITLES
            all_page_keys.append(item.page_key)
    assert len(all_page_keys) >= len(set(all_page_keys))


def test_person_analysis_is_the_first_workflow():
    first_mode = next(iter(NAVIGATION_MODES))
    items = NAVIGATION_MODES[AppMode.PERSON_ANALYSIS].items

    assert first_mode == AppMode.PERSON_ANALYSIS
    assert NAVIGATION_MODES[first_mode].title == "Person Analysis"
    assert len(items) == 1
    assert items[0].page_key == "person_analysis"


def test_privateframe_remains_a_separate_workflow():
    items = NAVIGATION_MODES[AppMode.PRIVATE_FRAME].items

    assert NAVIGATION_MODES[AppMode.PRIVATE_FRAME].title == "PrivateFrame"
    assert len(items) == 1
    assert items[0].page_key == "private_frame"


def test_face_verification_mode_is_focused():
    titles = [item.title for item in NAVIGATION_MODES[AppMode.FACE_VERIFICATION].items]
    keys = [item.page_key for item in NAVIGATION_MODES[AppMode.FACE_VERIFICATION].items]

    assert NAVIGATION_MODES[AppMode.FACE_VERIFICATION].title == "Face Recognition"
    assert titles == ["Face Recognition"]
    assert keys == ["verification"]


def test_face_swap_mode_is_single_workspace():
    items = NAVIGATION_MODES[AppMode.FACE_SWAP].items

    assert len(items) == 1
    assert items[0].title == "Face Swap"
    assert items[0].page_key == "image_face_swap"


def test_album_mode_is_single_workspace():
    items = NAVIGATION_MODES[AppMode.ALBUM_MANAGEMENT].items

    assert len(items) == 1
    assert items[0].title == "Album"
    assert items[0].page_key == "album"


def test_enterprise_mode_is_single_evaluation_workspace():
    items = NAVIGATION_MODES[AppMode.ENTERPRISE_EVALUATION].items

    assert len(items) == 1
    assert items[0].title == "Enterprise Evaluation"
    assert items[0].page_key == "enterprise_evaluation"


def test_mode_persistence(tmp_path):
    cfg = AppConfig(workspace_path=str(tmp_path))
    cfg.ui_last_mode = AppMode.FACE_SWAP.value
    cfg.ui_last_page_face_swap = "image_face_swap"
    save_config(cfg)

    loaded, exists = load_config(tmp_path / "config.json")

    assert exists is True
    assert loaded.ui_last_mode == AppMode.FACE_SWAP.value
    assert loaded.ui_last_page_face_swap == "image_face_swap"


def test_new_config_starts_with_person_analysis(tmp_path):
    cfg = AppConfig(workspace_path=str(tmp_path))

    assert cfg.ui_default_mode == AppMode.PERSON_ANALYSIS.value
    assert cfg.ui_last_mode == AppMode.PERSON_ANALYSIS.value
    assert cfg.ui_last_page_person_analysis == "person_analysis"
    assert cfg.ui_last_page_private_frame == "private_frame"


def test_existing_workspace_keeps_saved_mode_and_model(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "workspace_path": str(tmp_path),
        "model_name": "raccoon_l",
        "ui_default_mode": "private_frame",
        "ui_last_mode": "private_frame",
        "ui_last_page_private_frame": "private_frame",
    }), encoding="utf-8")

    cfg, exists = load_config(config_path)

    assert exists
    assert cfg.model_name == "raccoon_l"
    assert cfg.ui_default_mode == "private_frame"
    assert cfg.ui_last_mode == "private_frame"
    assert cfg.ui_last_page_person_analysis == "person_analysis"


@pytest.mark.parametrize("value", [None, "unknown", "person_analysis", "Person Analysis", AppMode.PERSON_ANALYSIS])
def test_person_analysis_mode_is_the_default_fallback(value):
    assert mode_from_value(value) == AppMode.PERSON_ANALYSIS


def test_default_detection_size_is_auto(tmp_path):
    cfg = AppConfig(workspace_path=str(tmp_path))

    assert cfg.det_size == [0, 0]
    assert cfg.det_size_label == "Auto"
    assert cfg.recognition_threshold == 0.4


def test_threshold_slider_default_is_028():
    import pytest

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.widgets.threshold_slider import ThresholdSlider

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    slider = ThresholdSlider()
    assert slider.value() == 0.4


def test_qt_plugin_paths_are_configured(monkeypatch):
    from pathlib import Path

    from insightface.gui.app import configure_qt_plugin_paths

    monkeypatch.delenv("QT_PLUGIN_PATH", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM_PLUGIN_PATH", raising=False)
    configure_qt_plugin_paths()

    assert Path(os.environ["QT_PLUGIN_PATH"]).exists()
    assert Path(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]).exists()
