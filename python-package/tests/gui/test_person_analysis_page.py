"""Person demo UI uses current-frame matches, not tracks or stored events."""
import os
from types import SimpleNamespace
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def page(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.person_analysis_page import PersonAnalysisPage
    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path / "workspace"),
        model_name="cheetah_s", model_root=str(tmp_path / "models"), ui_language="en")
    view = PersonAnalysisPage(SimpleNamespace(config=config))
    view._test_app = app
    yield view
    view.timer.stop()
    view.close()
    # close() only hides a widget. Destroy this test's Qt children while the
    # application is alive, rather than leaving parent cycles for Python GC.
    from PySide6.QtCore import QCoreApplication, QEvent
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def test_input_selection_and_reference_options(page, tmp_path):
    assert page.input_kind.currentData() == "video" and not page.start_button.isEnabled()
    page.input_kind.setCurrentIndex(1)
    assert page.start_button.isEnabled() and page.video_input.isHidden()
    page.input_kind.setCurrentIndex(2)
    assert not page.start_button.isEnabled()
    page.rtsp_url.setText("rtsp://user:password@example.test/live")
    assert page.start_button.isEnabled()
    page.add_reference_paths([tmp_path / "Alice.jpg", tmp_path / "Alice.jpg", tmp_path / "Alice-2.jpg"])
    page.references_table.item(1, 0).setText("Alice")
    assert page.references_table.rowCount() == 2
    assert [name for name, _ in page._job().references] == ["Alice", "Alice"]
    assert page.auto_update.isChecked() and page._job().auto_update
    page.auto_update.setChecked(False)
    assert not page._job().auto_update
    page.context.config.person_config = {"reid_similarity_threshold": .8}
    selected = page._job()
    page.context.config.person_config["reid_similarity_threshold"] = .9
    assert selected.person_config == {"reid_similarity_threshold": .8}
    assert not hasattr(page, "database_button") and not hasattr(page, "export_button")


@pytest.fixture
def reference_drop_inputs(tmp_path):
    from PIL import Image
    from PySide6.QtCore import QUrl

    photos = [tmp_path / "Alice.JPG", tmp_path / "张三.png"]
    for path in photos:
        Image.new("RGB", (12, 12), color="white").save(path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a reference photo")
    directory = tmp_path / "folder.jpg"
    directory.mkdir()
    invalid_urls = [QUrl.fromLocalFile(str(path)) for path in
                    (video, directory, tmp_path / "missing.jpg")]
    invalid_urls.append(QUrl("https://example.test/remote.jpg"))
    return photos, invalid_urls


def _send_reference_drop(page, target, urls=(), *, text=None):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

    page.resize(1200, 1000)
    page.show()
    page._test_app.processEvents()
    mime = QMimeData()
    if urls:
        mime.setUrls(urls)
    if text is not None:
        mime.setText(text)
    events = [
        QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier),
        QDragMoveEvent(QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier),
        QDropEvent(QPointF(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier),
    ]
    accepted = []
    for event in events:
        event.ignore()
        page._test_app.sendEvent(target, event)
        accepted.append(event.isAccepted())
    return accepted


@pytest.mark.parametrize("surface", ["references_group", "references_table",
                                     "add_references_button", "auto_update"])
def test_reference_drop_imports_valid_photos_across_add_people_area(page, reference_drop_inputs, surface):
    from PySide6.QtCore import Qt, QUrl

    photos, invalid_urls = reference_drop_inputs
    target = getattr(page, surface)
    if surface == "references_table":
        target = target.viewport()
    urls = [QUrl.fromLocalFile(str(photos[0])), *invalid_urls, QUrl.fromLocalFile(str(photos[1]))]
    assert _send_reference_drop(page, target, urls) == [True, True, True]
    assert page.references_table.rowCount() == 2
    assert [page.references_table.item(row, 0).text() for row in range(2)] == ["Alice", "张三"]
    assert [page.references_table.item(row, 1).data(Qt.UserRole) for row in range(2)] == [
        str(path.resolve()) for path in photos]


def test_repeated_reference_drop_keeps_edited_names_and_deduplicates(page, reference_drop_inputs):
    from PySide6.QtCore import QUrl

    photos, _ = reference_drop_inputs
    first = QUrl.fromLocalFile(str(photos[0]))
    second = QUrl.fromLocalFile(str(photos[1]))
    target = page.references_table.viewport()
    assert _send_reference_drop(page, target, [first, first]) == [True, True, True]
    assert page.references_table.rowCount() == 1
    page.references_table.item(0, 0).setText("Alice Chen")
    assert _send_reference_drop(page, target, [first, second, first, second]) == [True, True, True]
    assert page.references_table.rowCount() == 2
    assert [page.references_table.item(row, 0).text() for row in range(2)] == ["Alice Chen", "张三"]


@pytest.mark.parametrize("payload", ["video", "directory", "missing", "remote", "text"])
def test_reference_drop_rejects_unsupported_payloads(page, reference_drop_inputs, payload):
    photos, invalid_urls = reference_drop_inputs
    if payload == "text":
        accepted = _send_reference_drop(page, page.references_group, text=str(photos[0]))
    else:
        index = ["video", "directory", "missing", "remote"].index(payload)
        accepted = _send_reference_drop(page, page.references_group, [invalid_urls[index]])
    assert accepted == [False, False, False]
    assert page.references_table.rowCount() == 0


@pytest.mark.parametrize("state", ["running", "disabled", "unsupported_model"])
def test_reference_drop_rejects_when_controls_unavailable(page, reference_drop_inputs, state):
    from PySide6.QtCore import QUrl

    photos, _ = reference_drop_inputs
    if state == "running":
        page._running = True
    elif state == "disabled":
        page.references_group.setEnabled(False)
    else:
        page.context.config.model_name = "buffalo_l"
        page.refresh()
        assert not page.references_group.isEnabled()
    urls = [QUrl.fromLocalFile(str(path)) for path in photos]
    assert _send_reference_drop(page, page.references_table.viewport(), urls) == [False, False, False]
    assert page.references_table.rowCount() == 0


def test_invalid_source_or_wrong_package_cannot_start(page, monkeypatch):
    errors = []
    monkeypatch.setattr(page, "show_error", errors.append)
    page.input_kind.setCurrentIndex(2)
    page.rtsp_url.setText("https://example.test/video")
    page.start()
    assert errors and "RTSP" in errors[-1]
    assert not page._running
    page.context.config.model_name = "buffalo_l"
    page.input_kind.setCurrentIndex(1)
    assert not page.start_button.isEnabled()
    page.start()
    assert "cheetah" in errors[-1]


def test_model_test_job_blocks_a_second_person_job(page, monkeypatch):
    errors = []
    monkeypatch.setattr(page, "show_error", errors.append)
    page.context.person_analysis_jobs_in_progress = 1
    page.input_kind.setCurrentIndex(1)
    page.start()
    assert errors and "Wait" in errors[-1]
    assert page.runner is None
    assert not page._running


def test_face_only_and_unmatched_preview_without_anonymous_ids(page):
    from PySide6.QtWidgets import QGraphicsRectItem
    from PySide6.QtCore import Qt
    from insightface.gui.core.person_analysis import FramePreview
    face = SimpleNamespace(bbox=np.array([5, 15, 65, 85]))
    person = SimpleNamespace(body_bbox=None, face=face)
    match = SimpleNamespace(observation=person, person_id=None, matched_by=None, similarity=None)
    result = FramePreview([match], 7, 240, "video")
    page.preview.set_result(np.zeros((100, 160, 3), np.uint8), result, "en")
    texts = [x.text() for x in page.preview.scene.items() if hasattr(x, "text")]
    assert texts == ["Unmatched"]
    boxes = [x for x in page.preview.scene.items() if isinstance(x, QGraphicsRectItem) and x.parentItem() is None]
    assert len(boxes) == 1 and boxes[0].pen().style() == Qt.DashLine
    assert boxes[0].pen().isCosmetic()
    # The next frame deletes this scene's items. Keep only assertion values,
    # not Python wrappers for native graphics objects owned by that scene.
    del boxes
    match.person_id = "张三"
    match.matched_by, match.similarity = 'face', .912
    page.preview.set_result(np.zeros((100, 160, 3), np.uint8), result, "en")
    assert [x.text() for x in page.preview.scene.items() if hasattr(x, "text")] == ["张三 · Face 0.91"]
    match.matched_by = 'body'
    page.preview.set_result(np.zeros((100, 160, 3), np.uint8), result, "zh")
    assert [x.text() for x in page.preview.scene.items() if hasattr(x, "text")] == ["张三 · 人体 0.91"]


def test_preview_reads_latest_match_and_updates_statistics(page):
    from insightface.gui.core.person_analysis import FramePreview
    face = SimpleNamespace(bbox=np.array([5, 15, 20, 30]))
    person = SimpleNamespace(body_bbox=np.array([1, 2, 40, 80]), face=face)
    result = FramePreview([SimpleNamespace(observation=person, person_id="Alice", matched_by="face", similarity=.9)], 3, 80, "video")
    page.runner = SimpleNamespace(take_preview=lambda: (np.zeros((100, 160, 3), np.uint8), result))
    page._poll()
    assert "Analyzed frames: 3" in page.stats_label.text() and "Matched: 1" in page.stats_label.text()
    assert page.preview.matches == result.matches


def test_unsupported_model_dims_processing_but_keeps_model_and_license_buttons(page):
    page.context.config.model_name = "buffalo_l"
    page.refresh()
    assert not page.operation_panel.isEnabled()
    assert all(effect.opacity() < 1 for effect in page.operation_opacities)
    assert page.models_button.isEnabled() and page.commercial_button.isEnabled()
    page.context.config.model_name = "cheetah_l"
    page.refresh()
    assert page.operation_panel.isEnabled() and all(effect.opacity() == 1 for effect in page.operation_opacities)


def test_stop_is_nonblocking_request_and_does_not_close_worker_resources(page):
    calls = []
    page.runner = SimpleNamespace(request_stop=lambda: calls.append("request"))
    page.worker = SimpleNamespace(cancel=lambda: calls.append("cancel"))
    page._running = True
    page.stop()
    assert calls == ["request", "cancel"]
    assert not page.stop_button.isEnabled()
    page._running = False


def test_worker_start_failure_restores_controls_and_activity_count(page, monkeypatch):
    from insightface.gui.app import context_activity_count
    errors = []
    monkeypatch.setattr(page, 'show_error', errors.append)
    def fail(*args, **kwargs):
        raise RuntimeError('cannot start worker')
    monkeypatch.setattr(page, 'run_task', fail)
    page.input_kind.setCurrentIndex(1)
    page.start()
    assert errors == ['cannot start worker'] and not page._running
    assert context_activity_count(page.context, 'person_analysis_jobs_in_progress') == 0
    assert page.start_button.isEnabled() and not page.stop_button.isEnabled()


def test_commercial_action_is_preserved(page, monkeypatch):
    from insightface.gui.pages import person_analysis_page as module
    calls = []
    monkeypatch.setattr(module, "open_insightface_url", lambda url, **kw: calls.append((url, kw)))
    page.open_commercial()
    assert calls[0][0] == module.ENTERPRISE_HELP_URL


def test_video_position_formatting():
    from insightface.gui.core.person_analysis import display_time
    assert display_time(62123, "video") == "00:01:02.123"
    assert display_time(None, "video") == "—"


def _parameters_dialog(page):
    from insightface.gui.pages.person_analysis_page import PersonParametersDialog
    return PersonParametersDialog(
        page.context.config.person_config, page.context.config.ui_language,
        page._save_person_parameters, page,
    )


def test_advanced_controls_follow_every_api_default_and_type(page):
    from dataclasses import asdict, fields
    from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox
    from insightface.app.person.config import PersonConfig
    from insightface.gui.pages.person_analysis_page import PERSON_PARAMETER_COPY

    dialog = _parameters_dialog(page)
    defaults = asdict(PersonConfig())
    assert len(defaults) == 16
    assert set(dialog.controls) == set(defaults) == set(PERSON_PARAMETER_COPY)
    assert "max_face_samples" not in dialog.controls
    assert dialog.defaults == defaults
    assert {name: control.value() for name, control in dialog.controls.items()} == defaults
    assert dialog.overrides() == {}
    for field in fields(PersonConfig):
        control = dialog.controls[field.name]
        assert type(control.value()) is type(defaults[field.name])
        if type(defaults[field.name]) is int:
            assert isinstance(control, QSpinBox)
            assert control.minimum() == (0 if field.name in ("face_det_size", "body_det_size") else 1)
        else:
            assert isinstance(control, QDoubleSpinBox)
            assert control.minimum() == 0 and control.maximum() == 1
        assert dialog.field_labels[field.name].text()
        assert dialog.help_labels[field.name].text() == control.toolTip()
    assert dialog.scroll.widgetResizable()
    for name, step in (("face_det_size", 32), ("body_det_size", 64)):
        assert dialog.controls[name].singleStep() == step
        assert dialog.controls[name].specialValueText() == ("Default (640)" if name == "face_det_size" else "Model default")
    dialog.resize(580, 420)
    dialog.show()
    page._test_app.processEvents()
    assert dialog.scroll.verticalScrollBar().maximum() > 0
    assert dialog.save_button.isVisible() and dialog.cancel_button.isVisible()
    dialog.reject()


def test_custom_advanced_values_persist_and_flow_to_next_job(page):
    import json
    from dataclasses import asdict
    from pathlib import Path
    from PySide6.QtWidgets import QDialog
    from insightface.app.person.config import PersonConfig
    from insightface.gui.core.config import load_config

    page.input_kind.setCurrentIndex(page.input_kind.findData("camera"))
    original_job = page._job()
    defaults = asdict(PersonConfig())
    custom = {name: value + 3 if type(value) is int else round(value + .001, 3)
              for name, value in defaults.items()}
    custom["body_det_size"] = 640
    custom["face_det_size"] = 320
    dialog = _parameters_dialog(page)
    for name, value in custom.items():
        dialog.controls[name].setValue(value)
    assert page.context.config.person_config == {}
    dialog.accept()
    assert dialog.result() == QDialog.Accepted
    assert page.context.config.person_config == custom
    assert page._job().person_config == custom
    assert asdict(PersonConfig(**page._job().person_config)) == custom
    assert original_job.person_config == {}
    config_path = Path(page.context.config.workspace_path) / "config.json"
    assert json.loads(config_path.read_text())["person_config"] == custom
    loaded, exists = load_config(config_path)
    assert exists and loaded.person_config == custom
    reopened = _parameters_dialog(page)
    assert {name: control.value() for name, control in reopened.controls.items()} == custom
    reopened.reject()


def test_cancel_and_restore_only_change_draft_until_saved(page):
    from pathlib import Path
    from insightface.gui.core.config import save_config

    original = {"cpu_threads": 7, "face_margin": .15, "body_det_size": 640, "face_det_size": 320}
    page.context.config.person_config = original.copy()
    config_path = save_config(page.context.config)
    before = config_path.read_bytes()
    dialog = _parameters_dialog(page)
    dialog.controls["cpu_threads"].setValue(9)
    dialog.restore_defaults()
    assert dialog.controls["body_det_size"].value() == 0
    assert dialog.controls["face_det_size"].value() == 0
    assert dialog.overrides() == {}
    assert page.context.config.person_config == original
    assert config_path.read_bytes() == before
    dialog.reject()
    assert page.context.config.person_config == original
    assert config_path.read_bytes() == before
    reopened = _parameters_dialog(page)
    assert reopened.controls["cpu_threads"].value() == 7
    assert reopened.controls["body_det_size"].value() == 640
    assert reopened.controls["face_det_size"].value() == 320
    reopened.restore_defaults()
    reopened.accept()
    assert page.context.config.person_config == {}
    assert Path(config_path).read_bytes() != before


def test_advanced_save_drops_explicit_defaults_and_keeps_only_changes(page):
    from dataclasses import asdict
    from insightface.app.person.config import PersonConfig

    page.context.config.person_config = asdict(PersonConfig())
    dialog = _parameters_dialog(page)
    dialog.controls["cpu_threads"].setValue(PersonConfig().cpu_threads + 1)
    dialog.accept()
    assert page.context.config.person_config == {"cpu_threads": PersonConfig().cpu_threads + 1}


def test_failed_advanced_save_keeps_settings_and_draft_for_retry(page, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from insightface.gui.pages import person_analysis_page as module

    page.context.config.person_config = {"cpu_threads": 7}
    config_path = module.save_config(page.context.config)
    before = config_path.read_bytes()
    real_save = module.save_config
    dialog = _parameters_dialog(page)
    dialog.controls["cpu_threads"].setValue(9)

    def fail_save(_config):
        raise OSError("settings folder is read-only")

    monkeypatch.setattr(module, "save_config", fail_save)
    dialog.accept()
    assert dialog.result() != QDialog.Accepted
    assert "read-only" in dialog.error_label.text() and not dialog.error_label.isHidden()
    assert dialog.controls["cpu_threads"].value() == 9
    assert page.context.config.person_config == {"cpu_threads": 7}
    assert config_path.read_bytes() == before
    monkeypatch.setattr(module, "save_config", real_save)
    dialog.accept()
    assert dialog.result() == QDialog.Accepted
    assert page.context.config.person_config == {"cpu_threads": 9}


def test_advanced_save_validates_before_persistence(page, monkeypatch):
    from insightface.gui.pages import person_analysis_page as module

    saved = []
    monkeypatch.setattr(module, "save_config", saved.append)
    dialog = _parameters_dialog(page)
    monkeypatch.setattr(dialog, "overrides", lambda: {"cpu_threads": 0})
    dialog.accept()
    assert not saved and page.context.config.person_config == {}
    assert "positive integer" in dialog.error_label.text()


@pytest.mark.parametrize("name, multiple", [("face_det_size", 32), ("body_det_size", 64)])
def test_detection_size_rejects_non_multiple_before_save(page, monkeypatch, name, multiple):
    from PySide6.QtWidgets import QDialog
    from insightface.gui.pages import person_analysis_page as module

    saved = []
    monkeypatch.setattr(module, "save_config", saved.append)
    dialog = _parameters_dialog(page)
    dialog.controls[name].setValue(321)
    dialog.accept()
    assert dialog.result() != QDialog.Accepted
    assert not saved and page.context.config.person_config == {}
    assert name in dialog.error_label.text()
    assert str(multiple) in dialog.error_label.text()
    assert not dialog.error_label.isHidden()
    assert dialog.controls[name].value() == 321


def test_analysis_frequency_shared_by_all_sources_and_persists(page, tmp_path):
    from pathlib import Path
    from insightface.gui.core.config import load_config

    assert page.input_kind.currentData() == "video"
    assert not page.analysis_rate_controls.isHidden()
    assert page.analysis_max_fps.value() == 0
    assert page.analysis_max_fps.specialValueText() == "Auto"
    assert page.analysis_max_fps.objectName() == "personAnalysisMaxFps"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source existence is enough when creating a job")
    page.video_input.set_path(str(video))
    page.rtsp_url.setText("rtsp://example.test/live")
    page.analysis_max_fps.setValue(7.5)
    page.analysis_max_fps.editingFinished.emit()
    assert page.context.config.person_analysis_max_fps == 7.5
    for kind in ("video", "camera", "rtsp", "video"):
        page.input_kind.setCurrentIndex(page.input_kind.findData(kind))
        assert not page.analysis_rate_controls.isHidden()
        assert page.analysis_max_fps.value() == 7.5
        assert page._job().analysis_max_fps == 7.5
        assert page._job().source_kind == kind
    loaded, exists = load_config(Path(page.context.config.workspace_path) / "config.json")
    assert exists and loaded.person_analysis_max_fps == 7.5
    page.analysis_max_fps.setValue(0)
    page.analysis_max_fps.editingFinished.emit()
    assert page.context.config.person_analysis_max_fps == 0 and page._job().analysis_max_fps == 0
    page.input_kind.setCurrentIndex(page.input_kind.findData("camera"))
    assert page.analysis_max_fps.value() == 0 and page._job().analysis_max_fps == 0


def test_analysis_frequency_save_failure_restores_current_settings(page, monkeypatch):
    from insightface.gui.pages import person_analysis_page as module

    errors = []
    monkeypatch.setattr(page, "show_error", errors.append)
    def fail_save(_config):
        raise OSError("no space")
    monkeypatch.setattr(module, "save_config", fail_save)
    page.analysis_max_fps.setValue(10)
    page.analysis_max_fps.editingFinished.emit()
    assert errors and "no space" in errors[0]
    assert page.analysis_max_fps.value() == page.context.config.person_analysis_max_fps == 0


def test_advanced_and_frequency_controls_disabled_during_run_and_unsupported_model(page, monkeypatch):
    from insightface.gui.pages import person_analysis_page as module

    page.input_kind.setCurrentIndex(page.input_kind.findData("camera"))
    monkeypatch.setattr(page, "run_task", lambda *args, **kwargs: SimpleNamespace(cancel=lambda: None))
    opened = []
    monkeypatch.setattr(module, "PersonParametersDialog", lambda *args: opened.append(args))
    assert page.advanced_button.isEnabled() and page.analysis_max_fps.isEnabled()
    page.start()
    assert page._running
    assert not page.advanced_button.isEnabled() and not page.analysis_max_fps.isEnabled()
    page.open_advanced_parameters()
    assert not opened
    page._finished()
    assert page.advanced_button.isEnabled() and page.analysis_max_fps.isEnabled()
    page.context.config.model_name = "buffalo_l"
    page.refresh()
    assert not page.advanced_button.isEnabled() and not page.analysis_max_fps.isEnabled()
    page.open_advanced_parameters()
    assert not opened


def test_advanced_button_stays_visible_outside_input_scroll(page):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QScrollArea

    page.input_kind.setCurrentIndex(page.input_kind.findData("camera"))
    page.resize(1360, 920)
    page.show()
    page._test_app.processEvents()
    button = page.advanced_button
    position = button.mapTo(page, QPoint())
    assert button.isVisible()
    assert page.rect().contains(button.rect().translated(position))
    for scroll in page.findChildren(QScrollArea):
        assert not scroll.isAncestorOf(button)
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    page._test_app.processEvents()
    assert button.mapTo(page, QPoint()) == position
