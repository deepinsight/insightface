import pytest
import os
from pathlib import Path
import numpy as np


def test_main_window_smoke(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    PySide6 = pytest.importorskip("PySide6")
    from PySide6.QtCore import QEvent, QUrl, Qt
    from PySide6.QtWidgets import QAbstractItemView, QApplication, QLabel, QPushButton

    from insightface.gui.app import StudioContext, configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.face_engine import FaceEngine
    from insightface.gui.core.navigation import AppMode
    from insightface.gui.core.storage import Storage
    from insightface.gui.dialogs.license_dialog import LicenseDialog
    from insightface.gui.dialogs.model_manager_dialog import ModelManagerDialog
    from insightface.gui.dialogs.settings_dialog import SettingsDialog
    from insightface.gui.main_window import MainWindow
    from insightface.gui.resources import APP_ID, APP_PROCESS_NAME, app_icon_path, configure_application_metadata

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    configure_application_metadata(app)
    assert app.applicationName() == "InsightFace Evaluation Studio"
    assert app.organizationName() == "InsightFace"
    assert app.organizationDomain() == "insightface.ai"
    assert app.applicationVersion() == "2.1"
    assert APP_ID == "ai.insightface.evaluationstudio"
    assert APP_PROCESS_NAME == "InsightFace Evaluation Studio"
    assert app_icon_path().exists()
    assert not app.windowIcon().isNull()
    cfg = AppConfig(
        workspace_path=str(tmp_path),
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
        safe_mode=True,
        recognition_threshold=0.61,
        ui_language="en",
    )
    storage = Storage(cfg.database_path)
    engine = FaceEngine(model_name=cfg.model_name)
    window = MainWindow(StudioContext(cfg, True, storage, engine, str(tmp_path / "app.log")))
    window.show()
    assert window.mode_rail.isVisible()
    assert window.mode_rail.width() >= 220
    assert window.mode_list.count() == len(AppMode)
    assert window.mode_list.currentItem().data(Qt.UserRole) == AppMode.PERSON_ANALYSIS.value
    person_page = window.page_registry.get("person_analysis")
    assert window.stack.currentWidget() is person_page
    assert person_page.objectName() == "personAnalysisPage"
    assert window.page_registry.get("person_analysis") is person_page
    assert not person_page.stop_button.isEnabled()
    assert hasattr(person_page, "references_table")
    assert hasattr(person_page, "preview")
    assert not hasattr(person_page, "database_button")
    assert not hasattr(person_page, "event_table")
    assert not hasattr(person_page, "export_button")
    assert person_page.auto_update.isChecked()
    enterprise_help_button = window.findChild(QPushButton, "enterpriseHelpButton")
    assert enterprise_help_button is not None
    assert enterprise_help_button.text() == "Enterprise Help"
    assert "contact page" in enterprise_help_button.toolTip()
    local_notice = window.findChild(QLabel, "localProcessingNotice")
    assert local_notice is not None
    assert "All processing is local" in local_notice.text()
    assert "No images, embeddings, or reports are uploaded automatically" in local_notice.text()
    privateframe_page = window.page_registry.get("private_frame")
    window.change_mode(AppMode.PRIVATE_FRAME)
    assert window.stack.currentWidget() is privateframe_page
    assert privateframe_page.objectName() == "privateFramePage"
    assert privateframe_page.progress_bar.value() == 0
    assert not privateframe_page.start_button.isEnabled()
    cfg.model_name = "raccoon_s"
    window._model_configuration_changed()
    assert privateframe_page.start_button.isEnabled()
    assert not privateframe_page.cancel_button.isEnabled()
    window.change_mode(AppMode.FACE_VERIFICATION)
    window.open_page("verification")
    verification_page = window.page_registry.get("verification")
    assert abs(window.context.config.recognition_threshold - 0.4) < 1e-9
    assert abs(verification_page.threshold.value() - 0.4) < 1e-9
    notice = verification_page.findChild(QLabel, "noticeLabel")
    assert notice is not None
    assert "Gallery face embeddings are cached in memory" in notice.text()
    assert "All processing is local by default" not in notice.text()
    assert verification_page.multi_face_policy.objectName() == "verificationMultiFacePolicy"
    assert verification_page.multi_face_policy.currentText() == "Use largest centered face"
    assert "largest centered face" in notice.text()
    assert hasattr(verification_page.result_table, "_proportional_table_sizer")
    assert verification_page.result_table.selectionBehavior() == QAbstractItemView.SelectRows
    assert verification_page.result_table.selectionMode() == QAbstractItemView.SingleSelection
    verification_page._gallery_embedding_cache_key = ("old.jpg",)
    verification_page._gallery_embedding_cache = [{"path": "old.jpg"}]
    verification_page.set_gallery_paths(["new.jpg"])
    assert verification_page._gallery_embedding_cache_key is None
    assert verification_page._gallery_embedding_cache is None
    verification_page._gallery_embedding_cache_key = ("use_largest_face", "new.jpg")
    verification_page._gallery_embedding_cache = [{"path": "new.jpg"}]
    verification_page.results = [{"path": "new.jpg"}]
    verification_page.result_table.setRowCount(1)
    verification_page.multi_face_policy.setCurrentText("Mark as skip")
    assert verification_page._gallery_embedding_cache_key is None
    assert verification_page._gallery_embedding_cache is None
    assert verification_page.result_table.rowCount() == 0
    assert "multi-face query stops" in notice.text()
    label_texts = [label.text() for label in verification_page.findChildren(QLabel)]
    assert "Mode: waiting for gallery" not in label_texts
    for button in verification_page.findChildren(QPushButton):
        assert button.toolTip()
    face_swap_page = window.page_registry.get("image_face_swap")
    assert face_swap_page.output_view.objectName() == "imageViewer"
    assert face_swap_page.output_view.viewport().objectName() == "imageViewerViewport"
    album_page = window.page_registry.get("album")
    assert abs(album_page.cluster_threshold.value() - 0.48) < 1e-9
    album_threshold_help = album_page.findChild(QLabel, "albumThresholdHelp")
    assert album_threshold_help is not None
    assert "cosine distance = 1 - cosine threshold" in album_threshold_help.text()
    assert not hasattr(album_page, "match_threshold")
    assert album_page.min_samples.value() == 2
    assert album_page.min_face_size.value() == 80
    assert album_page.algorithm_label.text().startswith("Algorithm: DBSCAN")
    assert hasattr(album_page, "import_button")
    assert hasattr(album_page, "rebuild_button")
    assert album_page.cluster_table.columnCount() == 2
    assert album_page.cluster_table.horizontalHeaderItem(0).text() == "Thumbnail"
    assert album_page.cluster_table.horizontalHeaderItem(1).text() == "Photos"
    assert album_page.cluster_table.selectionBehavior() == QAbstractItemView.SelectRows
    assert album_page.photo_table.selectionBehavior() == QAbstractItemView.SelectRows
    from insightface.gui.core.utils import encode_webp_thumbnail

    thumb = encode_webp_thumbnail(np.zeros((40, 80, 3), dtype=np.uint8), max_side=80, quality=70)
    assert thumb is not None
    assert album_page._icon_from_bytes(thumb, album_page.cluster_table.iconSize()) is not None
    album_page.clusters = [{"id": 1, "photo_count": 2, "face_count": 3, "name": "Album Person 1", "thumbnail_face_id": 7}]
    album_page.cluster_items = {1: [{"id": 7, "thumbnail": thumb, "media_path": str(tmp_path / "album.jpg")}]}
    album_page._populate_clusters()
    assert album_page.cluster_table.item(0, 0).textAlignment() == Qt.AlignCenter
    assert album_page.cluster_table.item(0, 1).textAlignment() == Qt.AlignCenter
    enterprise_page = window.page_registry.get("enterprise_evaluation")
    enterprise_cards = enterprise_page.findChildren(PySide6.QtWidgets.QFrame, "enterpriseCard")
    assert len(enterprise_cards) >= 3
    assert enterprise_page.dataset_root.acceptDrops()
    assert enterprise_page.dataset_drop_card.acceptDrops()
    assert not enterprise_page.dataset_root.select_button.isHidden()
    assert enterprise_page.dataset_root.remove_button.isHidden()
    assert not hasattr(enterprise_page, "dataset_path_status")
    class _Mime:
        def hasUrls(self):
            return True

        def urls(self):
            return [QUrl.fromLocalFile(str(tmp_path))]

    class _Event:
        def mimeData(self):
            return _Mime()

    assert enterprise_page._dataset_drop_path(_Event()) == str(tmp_path)
    class _DragEvent:
        def __init__(self, event_type):
            self._event_type = event_type
            self.accepted = False
            self.ignored = False

        def type(self):
            return self._event_type

        def mimeData(self):
            return _Mime()

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.ignored = True

    drag_event = _DragEvent(QEvent.DragEnter)
    assert enterprise_page.dataset_root.eventFilter(enterprise_page.dataset_root.path_label, drag_event)
    assert drag_event.accepted
    drop_event = _DragEvent(QEvent.Drop)
    assert enterprise_page.dataset_root.eventFilter(enterprise_page.dataset_root.path_label, drop_event)
    assert drop_event.accepted
    assert enterprise_page.dataset_root.path() == str(tmp_path)
    assert enterprise_page.dataset_root.path_label.text() == f"Selected folder:\n{tmp_path}"
    assert enterprise_page.dataset_root.path_label.toolTip() == str(tmp_path)
    assert enterprise_page.dataset_root.select_button.isHidden()
    assert not enterprise_page.dataset_root.remove_button.isHidden()
    enterprise_page.dataset_root.clear()
    assert enterprise_page.dataset_root.path() == ""
    assert not enterprise_page.dataset_root.select_button.isHidden()
    assert enterprise_page.dataset_root.remove_button.isHidden()
    card_drop_event = _DragEvent(QEvent.Drop)
    assert enterprise_page.eventFilter(enterprise_page.dataset_drop_card, card_drop_event)
    assert card_drop_event.accepted
    assert enterprise_page.dataset_root.path_label.text() == f"Selected folder:\n{tmp_path}"
    assert hasattr(enterprise_page, "help_button")
    assert enterprise_page.help_button.text() == "Dataset Rules"
    assert not hasattr(enterprise_page, "export_pdf_button")
    assert "Export PDF" not in [button.text() for button in enterprise_page.findChildren(QPushButton)]
    assert "Open Dataset Rules" in enterprise_page.mode_summary.text()
    assert "Require exactly one face" in enterprise_page.mode_summary.text()
    enterprise_page.auto_split.setChecked(False)
    assert "Auto Split is off" in enterprise_page.mode_summary.text()
    enterprise_page.eval_mode.setCurrentText("1:N Identification")
    assert "gallery/<identity>" in enterprise_page.mode_summary.text()
    enterprise_page.multi_face_policy.setCurrentText("Use largest centered face")
    assert "Use largest centered face" in enterprise_page.mode_summary.text()
    from insightface.gui.pages.enterprise_eval_page import DatasetRulesDialog

    rules_dialog = DatasetRulesDialog(window)
    assert rules_dialog.windowTitle() == "Evaluation Dataset Rules"
    rules_dialog.close()
    assert enterprise_page.output.minimumHeight() >= 260
    for mode in AppMode:
        window.change_mode(mode)
        assert window.mode_rail.isVisible()
        assert local_notice.isVisible()
        assert window.mode_list.currentItem().data(Qt.UserRole) == mode.value
        assert window.sidebar_list.count() > 0
        assert not window.sidebar.isVisible()
        sidebar_titles = [window.sidebar_list.item(i).text() for i in range(window.sidebar_list.count())]
        assert "Settings" not in sidebar_titles
        assert "Model Settings" not in sidebar_titles
        assert "Model Downloads" not in sidebar_titles
        assert "License Center" not in sidebar_titles
    settings_dialog = SettingsDialog(window.context, window)
    assert hasattr(settings_dialog, "theme")
    assert settings_dialog.theme.count() >= 7
    assert window.context.config.ui_theme == "azure_lab"
    assert settings_dialog.theme.currentText() == "Azure Lab"
    assert hasattr(settings_dialog, "language")
    assert settings_dialog.language.findData("system") >= 0
    assert settings_dialog.language.findData("zh") >= 0
    assert not hasattr(settings_dialog, "workspace")
    assert not hasattr(settings_dialog, "default_mode")
    model_dialog = ModelManagerDialog(window.context, window)
    assert model_dialog.minimumWidth() >= 1120
    model_dialog.open_page("Model Downloads")
    model_dialog.resize(1120, 700)
    model_dialog.show()
    app.processEvents()
    downloads = model_dialog.downloads_page
    assert downloads.table.geometry().bottom() < downloads.source_footer.geometry().top()
    assert hasattr(model_dialog, "run_task")
    assert model_dialog.downloads_page.table.selectionBehavior() == QAbstractItemView.SelectRows
    assert model_dialog.downloads_page.table.selectionMode() == QAbstractItemView.SingleSelection
    assert not hasattr(model_dialog.runtime_page, "threshold")
    assert not hasattr(model_dialog.runtime_page, "workers")
    assert not hasattr(model_dialog.runtime_page, "frame_interval")
    assert hasattr(model_dialog.runtime_page, "gfpgan_enabled")
    assert hasattr(model_dialog.runtime_page, "gfpgan_model_combo")
    assert not model_dialog.runtime_page.gfpgan_enabled.isEnabled()
    gfpgan_path = Path(cfg.model_root) / "models" / "GFPGANv1.4" / "GFPGANv1.4.onnx"
    gfpgan_path.parent.mkdir(parents=True)
    gfpgan_path.write_bytes(b"fake")
    model_dialog.runtime_page.refresh()
    assert model_dialog.runtime_page.gfpgan_enabled.isEnabled()
    license_dialog = LicenseDialog(window.context, window)
    license_buttons = [button.text() for button in license_dialog.findChildren(QPushButton)]
    assert license_buttons == ["Visit Homepage", "Contact Enterprise Support"]
    assert license_dialog.height() >= 660
    assert license_dialog.page.table.minimumHeight() >= 260
    assert license_dialog.page.findChild(QLabel, "noticeLabel") is None
    from insightface.gui.pages.license_center_page import ENTERPRISE_HELP_URL, HOMEPAGE_URL

    assert HOMEPAGE_URL == "https://www.insightface.ai"
    assert ENTERPRISE_HELP_URL == "https://www.insightface.ai/contact"
    dialogs = [settings_dialog, model_dialog, license_dialog]
    for dialog in dialogs:
        dialog.close()
    assert window.windowTitle().startswith("InsightFace Evaluation Studio")
    window.close()


@pytest.mark.parametrize(
    "mode,model", [("person_analysis", "buffalo_l"), ("face_verification", "cheetah_s"), ("face_verification", "cheetah_l")],
)
def test_person_workflow_and_packages_do_not_autoload_face_engine(tmp_path, monkeypatch, mode, model):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import StudioContext, configure_qt_plugin_paths, create_face_engine
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.storage import Storage
    from insightface.gui.main_window import MainWindow

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    cfg = AppConfig(workspace_path=str(tmp_path), model_root=str(tmp_path / "models"),
                    model_name=model, ui_last_mode=mode, auto_load_model=True)
    engine = create_face_engine(cfg)
    monkeypatch.setattr(engine, "load", lambda: pytest.fail("FaceEngine must stay unloaded"))
    window = MainWindow(StudioContext(cfg, True, Storage(cfg.database_path), engine, str(tmp_path / "app.log")))
    window._auto_load_model_if_needed()
    window.auto_load_model()
    app.processEvents()
    assert not window.active_workers
    assert not engine.is_loaded()
    window.close()


def test_person_analysis_blocks_model_manager_while_running(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from insightface.gui import main_window
    from insightface.gui.app import StudioContext, configure_qt_plugin_paths, create_face_engine
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.storage import Storage

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    cfg = AppConfig(workspace_path=str(tmp_path), model_root=str(tmp_path / "models"),
                    auto_load_model=False, ui_language="en")
    context = StudioContext(cfg, True, Storage(cfg.database_path), create_face_engine(cfg), str(tmp_path / "app.log"))
    window = main_window.MainWindow(context)
    context.person_analysis_jobs_in_progress = 1
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(main_window, "ModelManagerDialog", lambda *_args, **_kwargs: pytest.fail("Models must stay closed"))
    window.open_model_manager()
    assert len(warnings) == 1
    assert warnings[0][2] == "Stop Person Analysis before opening Models."
    context.person_analysis_jobs_in_progress = 0
    app.processEvents()
    window.close()


def test_album_clustering_does_not_match_people_library(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import StudioContext, configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.face_engine import FaceEngine
    from insightface.gui.core.storage import Storage
    from insightface.gui.pages.album_page import AlbumPage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    cfg = AppConfig(workspace_path=str(tmp_path), auto_load_model=False, safe_mode=True)
    storage = Storage(cfg.database_path)
    existing_person_id = storage.add_person("Existing Person")
    storage.add_person("Second Existing Person")
    storage.add_face_sample(existing_person_id, np.array([1.0, 0.0], dtype=np.float32))
    media_id = storage.add_media_item(str(tmp_path / "album.jpg"), "image")
    face_id = storage.add_media_face(media_id, np.array([1.0, 0.0], dtype=np.float32))
    face = storage.list_media_faces()[0]

    page = AlbumPage(StudioContext(cfg, True, storage, FaceEngine(), str(tmp_path / "app.log")))
    clusters, _algorithm = page._cluster_faces([face], cosine_threshold=0.48, min_samples=1)

    assert clusters[0]["source"] == "album"
    assert clusters[0]["id"] == 1
    assert clusters[0]["name"] == "Album Person 1"
    assert page.cluster_items[clusters[0]["id"]][0]["id"] == face_id
