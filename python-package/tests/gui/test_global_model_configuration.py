import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from insightface.gui.app import (
    create_face_engine,
    engine_matches_config,
    reconfigure_context_engine,
)
from insightface.gui.core.config import AppConfig, load_config
from insightface.gui.core.model_packages import (
    CUSTOM_MODEL_CHOICE,
    GUI_MODEL_PACKAGES,
    PRIVATEFRAME_MODEL_PACKAGES,
    inspect_privateframe_model,
    is_gui_model_package_asset,
)


def _write_v2_package(
    model_root: Path,
    name: str = "raccoon_s",
    *,
    tasks: tuple[str, ...] = ("detection", "verification", "recognition"),
) -> Path:
    package = model_root / "models" / name
    package.mkdir(parents=True)
    task_document = {}
    for task in tasks:
        filename = f"{task}.onnx"
        (package / filename).write_bytes(task.encode("utf-8"))
        task_document[task] = {"file": filename}
    (package / "MODEL.LICENSE").write_text("test", encoding="utf-8")
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 2,
                "model_id": name,
                "tasks": task_document,
                "license": "MODEL.LICENSE",
            }
        ),
        encoding="utf-8",
    )
    return package


def test_new_gui_config_defaults_to_cheetah_without_migrating_existing_json(
    tmp_path,
):
    fresh = AppConfig(workspace_path=str(tmp_path / "fresh"))
    assert fresh.model_name == "cheetah_s"

    existing_path = tmp_path / "existing" / "config.json"
    existing_path.parent.mkdir()
    existing_path.write_text(
        json.dumps(
            {
                "workspace_path": str(existing_path.parent),
                "model_name": "buffalo_l",
                "model_root": str(tmp_path / "legacy-root"),
            }
        ),
        encoding="utf-8",
    )

    loaded, exists = load_config(existing_path)

    assert exists is True
    assert loaded.model_name == "buffalo_l"
    assert loaded.model_root == str(tmp_path / "legacy-root")


def test_shared_gui_catalog_contains_privateframe_and_legacy_packages():
    assert tuple(GUI_MODEL_PACKAGES[:2]) == ("cheetah_s", "cheetah_l")
    assert tuple(GUI_MODEL_PACKAGES[2:4]) == ("raccoon_s", "raccoon_l")
    assert PRIVATEFRAME_MODEL_PACKAGES == {"raccoon_s", "raccoon_l"}
    assert {"buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"}.issubset(
        GUI_MODEL_PACKAGES
    )


@pytest.mark.parametrize("name", ["raccoon_s", "raccoon_l"])
def test_privateframe_missing_supported_package_allows_first_use_download(
    tmp_path,
    name,
):
    status = inspect_privateframe_model(name, tmp_path)

    assert status.state == "missing"
    assert status.can_start is True
    assert status.package_path == tmp_path / "models" / name
    assert "first use" in status.message


def test_privateframe_rejects_non_raccoon_and_invalid_installed_package(tmp_path):
    unsupported = inspect_privateframe_model("buffalo_l", tmp_path)
    assert unsupported.state == "unsupported"
    assert unsupported.can_start is False

    (tmp_path / "models" / "raccoon_s").mkdir(parents=True)
    invalid = inspect_privateframe_model("raccoon_s", tmp_path)
    assert invalid.state == "invalid"
    assert invalid.can_start is False
    assert "manifest is missing" in invalid.message


def test_privateframe_v2_task_requirements_follow_recognition_policy(tmp_path):
    package = _write_v2_package(
        tmp_path,
        tasks=("detection", "verification"),
    )

    redaction_only = inspect_privateframe_model("raccoon_s", tmp_path)
    selective = inspect_privateframe_model(
        "raccoon_s",
        tmp_path,
        require_recognition=True,
    )

    assert redaction_only.state == "ready"
    assert redaction_only.package_path == package
    assert selective.state == "invalid"
    assert selective.can_start is False
    assert "recognition" in selective.message


def test_privateframe_rejects_manifest_artifact_sha_mismatch(tmp_path):
    package = _write_v2_package(tmp_path)
    verifier = package / "verification.onnx"
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"]["verification"]["sha256"] = hashlib.sha256(
        verifier.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert inspect_privateframe_model("raccoon_s", tmp_path).state == "ready"
    verifier.write_bytes(b"corrupt-verifier-artifact")

    invalid = inspect_privateframe_model("raccoon_s", tmp_path)
    assert invalid.state == "invalid"
    assert invalid.can_start is False
    assert "SHA-256 mismatch" in invalid.message


def test_catalog_model_assets_are_the_only_global_download_choices():
    assert is_gui_model_package_asset(
        name="raccoon_s.zip",
        source="InsightFace",
    )
    assert is_gui_model_package_asset(
        name="buffalo_l.zip",
        source="insightface",
    )
    assert not is_gui_model_package_asset(
        name="external.zip",
        source="InsightFace",
    )
    assert not is_gui_model_package_asset(
        name="raccoon_s.zip",
        source="third party",
    )
    assert not is_gui_model_package_asset(
        name="inswapper_128.onnx",
        source="InsightFace",
    )


def test_engine_builder_ignores_stale_custom_dir_for_builtin_model(tmp_path):
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(tmp_path / "root"),
        custom_model_dir=str(tmp_path / "stale-custom"),
        provider="CPU",
    )

    engine = create_face_engine(config)

    assert engine.model_name == "raccoon_s"
    assert engine.custom_model_dir is None
    assert engine_matches_config(engine, config)


def test_context_engine_is_rebuilt_only_when_global_configuration_changes(tmp_path):
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="buffalo_l",
        model_root=str(tmp_path / "root"),
        provider="CPU",
    )
    context = SimpleNamespace(config=config, engine=create_face_engine(config))
    original = context.engine

    assert reconfigure_context_engine(context) is False
    assert context.engine is original

    config.model_name = "raccoon_s"
    assert reconfigure_context_engine(context) is True
    assert context.engine is not original
    assert context.engine.model_name == "raccoon_s"
    assert reconfigure_context_engine(context) is False


def test_runtime_settings_apply_catalog_and_custom_model_semantics(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.pages.model_settings_page import ModelSettingsPage

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(tmp_path / "models-a"),
        custom_model_dir=str(tmp_path / "stale-custom"),
        provider="CPU",
    )

    class Engine:
        @staticmethod
        def get_runtime_info():
            return {}

        @staticmethod
        def is_loaded():
            return False

    page = ModelSettingsPage(SimpleNamespace(config=config, engine=Engine()))

    assert page.model_combo.currentData() == "raccoon_s"
    assert page.custom_dir.isHidden()
    page.model_combo.setCurrentIndex(page.model_combo.findData("raccoon_l"))
    page.model_root.setText(str(tmp_path / "models-b"))
    page.model_root.editingFinished.emit()
    assert page.model_combo.currentData() == "raccoon_l"
    page.save()
    assert config.model_name == "raccoon_l"
    assert config.custom_model_dir == ""
    assert config.model_root == str(tmp_path / "models-b")

    page.model_combo.setCurrentIndex(page.model_combo.findData(CUSTOM_MODEL_CHOICE))
    page.custom_dir.setText("")
    with pytest.raises(ValueError, match="custom model directory"):
        page._apply_to_config()
    custom = str(tmp_path / "my-model")
    page.custom_dir.setText(custom)
    page._apply_to_config()
    assert config.model_name == custom
    assert config.custom_model_dir == custom
    page.close()


def test_first_launch_wizard_uses_shared_catalog_and_current_selection(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.main_window import FirstLaunchWizard

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path),
        model_name="buffalo_l",
        provider="CPU",
    )
    wizard = FirstLaunchWizard(config)

    choices = [wizard.model.itemData(index) for index in range(wizard.model.count())]
    assert choices == [*GUI_MODEL_PACKAGES, CUSTOM_MODEL_CHOICE]
    assert wizard.model.currentData() == "buffalo_l"
    assert wizard.provider.currentText() == "CPU"
    wizard.close()


def test_model_manager_separates_configuration_from_status_and_file_events(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.model_downloads import ModelAsset
    from insightface.gui.dialogs.model_manager_dialog import ModelManagerDialog
    from insightface.gui.pages import model_settings_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
        provider="CPU",
    )

    class Engine:
        def __init__(self, *_args, **_kwargs):
            self.loaded = False

        def load(self):
            self.loaded = True

        def is_loaded(self):
            return self.loaded

        @staticmethod
        def get_runtime_info():
            return {}

        @staticmethod
        def warmup():
            return {"warmup_ms": 1.0}

    active_engine = Engine()
    active_engine.loaded = True
    context = SimpleNamespace(config=config, engine=active_engine)
    dialog = ModelManagerDialog(context)
    configuration_events = []
    file_events = []
    dialog.modelChanged.connect(lambda: configuration_events.append("config"))
    dialog.modelFilesChanged.connect(lambda: file_events.append("files"))
    monkeypatch.setattr(model_settings_page, "FaceEngine", Engine)

    dialog.set_status("ordinary status")
    dialog.refresh_statusbar()
    dialog.runtime_page.test_load()
    dialog.runtime_page.warmup()
    assert configuration_events == []
    assert file_events == []

    dialog.refresh_model_pages()
    assert configuration_events == []
    assert file_events == ["files"]

    dialog.runtime_page.save()
    assert configuration_events == ["config"]

    dialog.downloads_page.assets = [
        ModelAsset(
            name="raccoon_l.zip",
            browser_download_url="https://example.invalid/raccoon_l.zip",
            source="InsightFace",
        )
    ]
    package_dir = Path(config.model_root) / "models" / "raccoon_l"
    package_dir.mkdir(parents=True)
    (package_dir / "detector.onnx").write_bytes(b"onnx")
    dialog.downloads_page.populate()
    dialog.downloads_page.table.selectRow(0)
    assert dialog.downloads_page.use_selected_button.isEnabled()
    assert not dialog.downloads_page.download_selected_button.isEnabled()
    dialog.downloads_page.use_selected_model()
    assert config.model_name == "raccoon_l"
    assert configuration_events == ["config", "config"]
    assert file_events == ["files", "files"]
    dialog.close()


def test_privateframe_page_reflects_global_compatibility_and_job_snapshot(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.pages.privateframe_page import (
        PrivateFramePage,
        build_privateframe_job,
    )

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="buffalo_l",
        model_root=str(tmp_path / "root-a"),
        auto_load_model=False,
    )
    page = PrivateFramePage(SimpleNamespace(config=config))

    assert page.model_label.text() == "buffalo_l"
    assert page.start_button.isEnabled() is False
    assert page.open_models_button.isEnabled() is False
    assert page.model_requirement_open_models_button.isEnabled()
    assert page.operation_panel.isEnabled() is False
    assert "raccoon_s or raccoon_l" in page.model_status_label.text()

    config.model_name = "raccoon_s"
    page.refresh()
    assert page.start_button.isEnabled()
    assert "first use" in page.model_status_label.text()

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    job = build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package=config.model_name,
        model_root=config.model_root,
        max_analysis_fps=15,
        redaction_method="gaussian",
        runtime_provider="CPU",
    )
    page._last_job = job
    page._set_running(True)
    config.model_name = "buffalo_l"
    config.model_root = str(tmp_path / "root-b")
    config.provider = "Auto"
    page.refresh()

    assert page.model_label.text() == "raccoon_s"
    assert page.model_root_label.text() == str((tmp_path / "root-a").resolve())
    assert "applies to the next run" in page.model_status_label.text()
    assert page.model_requirement_banner.isHidden()
    assert page.operation_panel.isEnabled()
    assert page.cancel_button.isEnabled()
    assert job.config_overrides["models.name"] == "raccoon_s"
    assert job.config_overrides["models.root"] == str((tmp_path / "root-a").resolve())
    assert job.config_overrides["runtime.provider"] == "CPUExecutionProvider"
    page._set_running(False)
    assert page.model_label.text() == "buffalo_l"
    assert page.start_button.isEnabled() is False
    assert page.model_requirement_banner.isHidden() is False
    assert page.operation_panel.isEnabled() is False
    page.close()


def test_verification_cache_is_engine_scoped_and_ignores_stale_worker_result(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.pages import verification_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])

    class Engine:
        def __init__(self, name, value):
            self.model_name = name
            self.value = value
            self.requested_providers = ["CPUExecutionProvider"]
            self.det_size = (640, 640)

        @staticmethod
        def is_loaded():
            return True

        def resolve_model_dir(self):
            return tmp_path / "models" / self.model_name

        def detect_faces(self, _image, source_path=None):
            del source_path
            return [
                SimpleNamespace(
                    normed_embedding=np.asarray(
                        [self.value, 1.0 - self.value],
                        dtype=np.float32,
                    ),
                    bbox=[0.0, 0.0, 8.0, 8.0],
                    det_score=0.9,
                )
            ]

    class TaskHost(QWidget):
        def __init__(self, context):
            super().__init__()
            self.context = context
            self.requests = []

        def run_task(self, title, fn, on_result=None, **_kwargs):
            self.requests.append((title, fn, on_result))
            return None

        def set_status(self, _message):
            pass

    old_engine = Engine("raccoon_s", 1.0)
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        provider="CPU",
        auto_load_model=False,
    )
    context = SimpleNamespace(config=config, engine=old_engine)
    host = TaskHost(context)
    page = verification_page.VerificationPage(context, parent=host)
    page.query_path = "query.jpg"
    page.query_image = np.zeros((16, 16, 3), dtype=np.uint8)
    page.gallery_paths = ["gallery.jpg"]
    monkeypatch.setattr(
        verification_page,
        "read_image",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )

    page.run_verification()
    _title, task, done = host.requests[-1]
    stale_payload = task()

    new_engine = Engine("raccoon_l", 0.0)
    context.engine = new_engine
    config.model_name = "raccoon_l"
    page.refresh()
    assert page._gallery_embedding_cache_key is None
    assert page.results == []

    done(stale_payload)
    assert page._gallery_embedding_cache_key is None
    assert page.results == []
    assert "global model changed" in page.status_label.text()

    page.run_verification()
    _title, task, done = host.requests[-1]
    done(task())
    assert page.results
    assert page._gallery_embedding_cache_key is not None
    assert page._gallery_embedding_cache_key[0] == page._engine_cache_identity(
        new_engine
    )
    page.close()
    host.close()


def test_model_download_activity_survives_dialog_close_and_blocks_privateframe(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.model_downloads import ModelAsset
    from insightface.gui.pages.model_download_page import ModelDownloadPage
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
    )
    context = SimpleNamespace(config=config)

    class TaskHost(QWidget):
        def __init__(self):
            super().__init__()
            self.requests = []

        def run_task(self, title, fn, on_result=None, **kwargs):
            self.requests.append((title, fn, on_result, kwargs))
            return object()

        def set_status(self, _message):
            pass

    host = TaskHost()
    downloads = ModelDownloadPage(context, parent=host)
    downloads.assets = [
        ModelAsset(
            name="raccoon_s.zip",
            browser_download_url="https://example.invalid/raccoon_s.zip",
            source="InsightFace",
        )
    ]
    downloads.populate()
    downloads.table.selectRow(0)
    downloads.download_selected()

    assert context.model_downloads_in_progress == 1
    assert len(host.requests) == 1
    downloads.close()

    privateframe = PrivateFramePage(context)
    privateframe.refresh()
    assert privateframe.start_button.isEnabled() is False
    errors = []
    monkeypatch.setattr(privateframe, "show_error", errors.append)
    privateframe.start_processing()
    assert errors == [
        "Wait for the model download to finish before starting PrivateFrame."
    ]
    assert getattr(context, "privateframe_jobs_in_progress", 0) == 0

    host.requests[0][3]["on_finished"]()
    assert context.model_downloads_in_progress == 0
    privateframe.refresh()
    assert privateframe.start_button.isEnabled()

    context.privateframe_jobs_in_progress = 1
    downloads.show()
    downloads._update_selection_actions()
    assert downloads.download_selected_button.isEnabled() is False
    download_errors = []
    monkeypatch.setattr(downloads, "show_error", download_errors.append)
    downloads.download_selected()
    assert download_errors == [
        "Wait for analysis processing to finish before downloading a model "
        "package."
    ]
    assert len(host.requests) == 1
    assert context.model_downloads_in_progress == 0

    context.privateframe_jobs_in_progress = 0
    privateframe.close()
    downloads.close()
    host.close()


def test_main_window_rejects_models_while_privateframe_is_running(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox

    from insightface.gui import main_window
    from insightface.gui.app import (
        StudioContext,
        configure_qt_plugin_paths,
        create_face_engine,
    )
    from insightface.gui.core.storage import Storage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
    )
    context = StudioContext(
        config,
        True,
        Storage(config.database_path),
        create_face_engine(config),
        str(tmp_path / "app.log"),
    )
    window = main_window.MainWindow(context)
    page = window.page_registry.get("private_frame")
    page._set_running(True)
    context.privateframe_jobs_in_progress = 1
    assert page.open_models_button.isEnabled() is False

    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )
    monkeypatch.setattr(
        main_window,
        "ModelManagerDialog",
        lambda *_args, **_kwargs: pytest.fail(
            "the model manager must not be constructed"
        ),
    )
    window.open_model_manager()

    assert len(warnings) == 1
    assert "PrivateFrame processing" in warnings[0][2]
    context.privateframe_jobs_in_progress = 0
    page._set_running(False)
    window.close()


def test_main_window_model_change_restores_privateframe_workspace(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui import main_window
    from insightface.gui.app import StudioContext, configure_qt_plugin_paths
    from insightface.gui.core.storage import Storage

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="buffalo_l",
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
    )
    context = StudioContext(
        config,
        True,
        Storage(config.database_path),
        create_face_engine(config),
        str(tmp_path / "app.log"),
    )
    window = main_window.MainWindow(context)
    page = window.page_registry.get("private_frame")

    assert page.model_requirement_banner.isHidden() is False
    assert page.operation_panel.isEnabled() is False

    config.model_name = "raccoon_s"
    window._model_configuration_changed()

    assert page.model_requirement_banner.isHidden()
    assert page.operation_panel.isEnabled()
    assert page.start_button.isEnabled()
    window.close()
