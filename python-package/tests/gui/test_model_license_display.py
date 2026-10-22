import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


from insightface.gui.core.config import AppConfig
from insightface.gui.core import licensing
from insightface.model_zoo.model_license import ModelLicenseInspection


def _inspection(package_path, status, message="license result"):
    return ModelLicenseInspection(
        package_path=package_path,
        model_id="raccoon_s",
        license_path=package_path / "MODEL.LICENSE",
        status=status,
        message=message,
    )


def test_license_model_directory_resolution_matches_gui_engine_rules(tmp_path):
    root = tmp_path / "root"
    stale_custom = tmp_path / "stale-custom"
    stale_custom.mkdir()
    builtin = AppConfig(
        workspace_path=str(tmp_path / "builtin-workspace"),
        model_name="raccoon_s",
        model_root=str(root),
        custom_model_dir=str(stale_custom),
    )
    assert licensing.resolve_configured_model_dir(builtin) == (
        root / "models" / "raccoon_s"
    )

    custom = tmp_path / "custom-model"
    custom.mkdir()
    custom_config = AppConfig(
        workspace_path=str(tmp_path / "custom-workspace"),
        model_name=str(custom),
        model_root=str(root),
        custom_model_dir=str(custom),
    )
    assert licensing.resolve_configured_model_dir(custom_config) == custom


def test_license_inspection_is_cached_and_invalidated_on_file_changes(
    tmp_path,
    monkeypatch,
):
    package = tmp_path / "root" / "models" / "raccoon_s"
    package.mkdir(parents=True)
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(tmp_path / "root"),
    )
    calls = []

    def inspect(model_dir, *, expected_model_id=None):
        calls.append((model_dir, expected_model_id))
        return _inspection(package, "default_non_commercial")

    monkeypatch.setattr(licensing, "inspect_model_package_license", inspect)
    licensing.invalidate_model_license_cache()

    first = licensing.model_license_display(config)
    second = licensing.model_license_display(config)

    assert first.status_text == "Research / Non-commercial"
    assert second.status_text == first.status_text
    assert calls == [(package, "raccoon_s")]

    (package / "MODEL.LICENSE").write_text("changed", encoding="utf-8")
    licensing.model_license_display(config)
    assert len(calls) == 2

    licensing.invalidate_model_license_cache()
    licensing.model_license_display(config)
    assert len(calls) == 3


def test_license_cache_key_does_not_raise_for_a_symlink_loop(tmp_path):
    package_loop = tmp_path / "package-loop"
    try:
        package_loop.symlink_to(package_loop.name)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name=str(package_loop),
        custom_model_dir=str(package_loop),
        model_root=str(tmp_path / "root"),
    )
    licensing.invalidate_model_license_cache()

    display = licensing.model_license_display(config)

    assert display.status_text == "Invalid model manifest"
    assert display.is_error is True


def test_unexpandable_model_root_does_not_crash_license_display(tmp_path):
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root="~insightface-user-that-must-not-exist",
    )
    licensing.invalidate_model_license_cache()

    display = licensing.model_license_display(config)

    assert display.status_text in {
        "Research / Non-commercial",
        "Invalid model manifest",
    }


@pytest.mark.parametrize(
    ("inspection_status", "display_status", "is_error"),
    [
        ("verified_non_commercial", "Research / Non-commercial", False),
        ("verified_commercial", "Commercial", False),
        ("default_non_commercial", "Research / Non-commercial", False),
        ("invalid", "Invalid model license", True),
        ("invalid_manifest", "Invalid model manifest", True),
        ("not_active", "Model license not active", True),
        ("expired", "Model license expired", True),
        ("dependency_missing", "License verification unavailable", True),
    ],
)
def test_license_inspection_statuses_have_user_facing_labels(
    tmp_path,
    monkeypatch,
    inspection_status,
    display_status,
    is_error,
):
    package = tmp_path / "root" / "models" / "raccoon_s"
    package.mkdir(parents=True)
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(tmp_path / "root"),
    )
    monkeypatch.setattr(
        licensing,
        "inspect_model_package_license",
        lambda *_args, **_kwargs: _inspection(
            package,
            inspection_status,
            message="specific inspection detail",
        ),
    )
    licensing.invalidate_model_license_cache()

    display = licensing.model_license_display(config)

    assert display.status_text == display_status
    assert display.is_error is is_error
    assert display.status_text != inspection_status
    if is_error:
        assert "specific inspection detail" in display.tooltip("en")
    else:
        assert "specific inspection detail" not in display.tooltip("en")


def test_main_window_and_license_center_refresh_without_loading_face_analysis(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import StudioContext, configure_qt_plugin_paths, create_face_engine
    from insightface.gui.core import face_engine
    from insightface.gui.core.storage import Storage
    from insightface.gui.dialogs.license_dialog import LicenseDialog
    from insightface.gui.main_window import MainWindow

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    # Test license refresh independently of platform font-dependent elision.
    monkeypatch.setattr(MainWindow, "_elide", lambda self, text, width: text)
    package = tmp_path / "model-root" / "models" / "raccoon_s"
    package.mkdir(parents=True)
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
        ui_language="en",
    )
    current_status = {"value": "invalid"}

    def inspect(*_args, **_kwargs):
        return _inspection(
            package,
            current_status["value"],
            message="bad signature",
        )

    monkeypatch.setattr(licensing, "inspect_model_package_license", inspect)
    monkeypatch.setattr(
        face_engine,
        "FaceAnalysis",
        lambda *_args, **_kwargs: pytest.fail(
            "license display must not create FaceAnalysis"
        ),
    )
    licensing.invalidate_model_license_cache()
    context = StudioContext(
        config,
        True,
        Storage(config.database_path),
        create_face_engine(config),
        str(tmp_path / "app.log"),
        runtime_safe_mode=True,
    )
    window = MainWindow(context)

    assert window.license_chip.text() == "Invalid model license"
    assert "bad signature" in window.license_chip.toolTip()
    assert not context.engine.is_loaded()
    dialog = LicenseDialog(context, window)
    assert "Invalid model license" in dialog.page.summary.text()
    assert dialog.page.model_license_detail.text() == (
        "The current model license is invalid."
    )

    current_status["value"] = "default_non_commercial"
    window._model_files_changed()
    dialog.page.refresh()

    assert window.license_chip.text() == "Research / Non-commercial"
    assert "Research / Non-commercial" in dialog.page.summary.text()
    assert not context.engine.is_loaded()
    dialog.close()
    window.close()


def test_model_license_visible_copy_is_translated_for_supported_languages():
    from insightface.gui.core.i18n import tr

    strings = (
        "Commercial",
        "Invalid model license",
        "Invalid model manifest",
        "Model license not active",
        "Model license expired",
        "License verification unavailable",
        "MODEL.LICENSE was not found. Non-commercial use is assumed by default.",
    )
    for language in ("zh", "ja", "ko", "es", "fr", "de", "pt", "ru"):
        for text in strings:
            assert tr(text, language) != text
