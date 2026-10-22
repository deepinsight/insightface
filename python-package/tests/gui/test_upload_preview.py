from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from insightface.gui.app import configure_qt_plugin_paths
from insightface.gui.widgets.image_viewer import ImageViewer
from insightface.gui.widgets.upload_preview import UploadPreview


@pytest.fixture(scope="module")
def app():
    configure_qt_plugin_paths()
    return QApplication.instance() or QApplication([])


def _settle(app):
    # Hiding a scroll bar can deliver another viewport resize event.
    for _ in range(3):
        app.processEvents()


def _wheel(widget, delta=120):
    local = QPointF(widget.rect().center())
    return QWheelEvent(
        local,
        QPointF(widget.mapToGlobal(local.toPoint())),
        QPoint(),
        QPoint(0, delta),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )


def _zoom(viewer, app):
    for _ in range(8):
        viewer.wheelEvent(_wheel(viewer.viewport()))
    _settle(app)
    assert viewer.horizontalScrollBar().maximum() > 0
    assert viewer.verticalScrollBar().maximum() > 0


def _assert_empty(viewer):
    assert viewer.image is None
    assert viewer.pixmap_item is None
    assert viewer.faces == []
    assert viewer.scene.items() == []
    assert viewer.scene.sceneRect().isEmpty()
    assert viewer.sceneRect().isEmpty()
    assert viewer.transform().isIdentity()
    assert viewer.dragMode() == QGraphicsView.NoDrag
    for bar in (viewer.horizontalScrollBar(), viewer.verticalScrollBar()):
        assert bar.minimum() == bar.maximum() == bar.value() == 0
        assert not bar.isVisible()


def _assert_fitted(viewer):
    image_rect = viewer.mapFromScene(viewer.pixmap_item.boundingRect()).boundingRect()
    assert image_rect.width() <= viewer.viewport().width()
    assert image_rect.height() <= viewer.viewport().height()
    assert viewer.horizontalScrollBar().maximum() == 0
    assert viewer.verticalScrollBar().maximum() == 0


@pytest.mark.parametrize("shape", [(1080, 1920, 3), (1920, 1080, 3)])
def test_clear_resets_zoom_scroll_ranges_and_historical_scene_bounds(app, shape):
    viewer = ImageViewer()
    viewer.resize(360, 240)
    viewer.show()
    _settle(app)
    try:
        for _ in range(2):
            viewer.set_image(np.zeros(shape, dtype=np.uint8))
            _settle(app)
            _assert_fitted(viewer)
            _zoom(viewer, app)
            viewer.setSceneRect(QRectF(-20, -30, 4000, 4000))
            viewer.horizontalScrollBar().setValue(100)
            viewer.verticalScrollBar().setValue(200)
            viewer.set_faces([{"bbox": [20, 30, 80, 90], "label": "Test face"}])

            viewer.set_image(None)
            _settle(app)
            _assert_empty(viewer)
            empty_wheel = _wheel(viewer.viewport())
            viewer.wheelEvent(empty_wheel)
            assert not empty_wheel.isAccepted()
            viewer.resize(420, 280)
            _settle(app)
            _assert_empty(viewer)

            # A much smaller next image must not inherit the previous extent.
            viewer.set_image(np.zeros((37, 81, 3), dtype=np.uint8))
            _settle(app)
            assert viewer.scene.sceneRect() == QRectF(0, 0, 81, 37)
            _assert_fitted(viewer)
            viewer.set_image(None)
    finally:
        viewer.close()


def test_loaded_image_keeps_zoom_pan_and_face_overlays(app):
    viewer = ImageViewer()
    viewer.resize(360, 240)
    viewer.show()
    viewer.set_image(np.zeros((1080, 1920, 3), dtype=np.uint8))
    _settle(app)
    try:
        viewer.set_faces([{"bbox": [700, 400, 1100, 700], "label": "Face"}])
        assert len(viewer.scene.items()) == 3
        clicked = []
        viewer.faceClicked.connect(clicked.append)
        point = viewer.mapFromScene(QPointF(900, 550))
        viewport = viewer.viewport()

        def mouse(kind, point, button, buttons):
            return QMouseEvent(
                kind,
                QPointF(point),
                QPointF(viewport.mapToGlobal(point)),
                button,
                buttons,
                Qt.NoModifier,
            )

        app.sendEvent(viewport, mouse(QEvent.MouseButtonPress, point, Qt.LeftButton, Qt.LeftButton))
        app.sendEvent(viewport, mouse(QEvent.MouseButtonRelease, point, Qt.LeftButton, Qt.NoButton))
        assert clicked == [0]
        _zoom(viewer, app)
        assert viewer.dragMode() == QGraphicsView.ScrollHandDrag
        viewer.centerOn(960, 540)
        start = viewport.rect().center()
        before = viewer.horizontalScrollBar().value()
        app.sendEvent(viewport, mouse(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
        end = start - QPoint(40, 20)
        app.sendEvent(viewport, mouse(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
        app.sendEvent(viewport, mouse(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        assert viewer.horizontalScrollBar().value() != before

        zoom = viewer.transform()
        viewer.resize(440, 300)
        _settle(app)
        assert viewer.transform() == zoom
        viewer.fit_to_window()
        _settle(app)
        _assert_fitted(viewer)
        assert len(viewer.scene.items()) == 3
    finally:
        viewer.close()


def test_preview_hides_empty_or_pending_thumbnail_and_refits_replacement(app, tmp_path):
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    preview = UploadPreview("Video", [".mp4"], "Videos (*.mp4)")
    preview.resize(400, 300)
    preview.show()
    _settle(app)
    try:
        assert preview.viewer.isHidden()
        assert preview.placeholder.isVisible()
        preview.set_path(str(first))
        _settle(app)
        assert preview.viewer.isHidden()
        _assert_empty(preview.viewer)
        preview.set_image(np.zeros((720, 1280, 3), dtype=np.uint8))
        _settle(app)
        assert preview.viewer.isVisible()
        assert preview.placeholder.isHidden()
        _assert_fitted(preview.viewer)
        _zoom(preview.viewer, app)

        preview.set_path(str(second))
        _settle(app)
        assert preview.path() == str(second)
        assert preview.viewer.isHidden()
        assert preview.placeholder.isVisible()
        _assert_empty(preview.viewer)
        preview.set_image(None)  # A failed/pending thumbnail retains the filename.
        assert preview.path() == str(second)
        assert preview.viewer.isHidden()
        preview.set_image(np.zeros((1280, 720, 3), dtype=np.uint8))
        _settle(app)
        _assert_fitted(preview.viewer)
        signals = []
        preview.removed.connect(lambda: signals.append(("removed",)))
        preview.pathChanged.connect(lambda path: signals.append(("pathChanged", path)))
        # A window-level event verifies hit testing too: an overlapping viewer
        # must not cover the floating remove button after a stack switch.
        remove_point = preview.remove_button.mapTo(preview, preview.remove_button.rect().center())
        QTest.mouseClick(preview.windowHandle(), Qt.LeftButton, Qt.NoModifier, remove_point)
        _settle(app)
        assert signals == [("removed",), ("pathChanged", "")]
        assert preview.path() == ""
        assert preview.viewer.isHidden()
        assert preview.placeholder.isVisible()
        assert preview.file_label.isHidden()
        assert preview.remove_button.isHidden()
        _assert_empty(preview.viewer)
    finally:
        preview.close()


@pytest.mark.parametrize("thumbnail_state", ["pending", "ready", "failed"])
def test_compact_preview_remove_button_is_above_active_stack_page(app, tmp_path, thumbnail_state):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"widget fixture only")
    preview = UploadPreview("Video", [".mp4"], "Videos (*.mp4)")
    preview.setFixedSize(120, 120)
    preview.show()
    _settle(app)
    try:
        preview.set_path(str(source), emit=False)
        if thumbnail_state != "pending":
            preview.set_image(np.zeros((40, 80, 3), dtype=np.uint8))
        if thumbnail_state == "failed":
            preview.set_image(None)
        _settle(app)
        signals = []
        preview.removed.connect(lambda: signals.append(("removed",)))
        preview.pathChanged.connect(lambda path: signals.append(("pathChanged", path)))
        point = preview.remove_button.mapTo(preview, preview.remove_button.rect().center())
        QTest.mouseClick(preview.windowHandle(), Qt.LeftButton, Qt.NoModifier, point)
        _settle(app)
        assert signals == [("removed",), ("pathChanged", "")]
        assert preview.path() == ""
        assert preview.viewer.isHidden()
        assert preview.placeholder.isVisible()
        assert preview.remove_button.isHidden()
        _assert_empty(preview.viewer)
    finally:
        preview.close()


def test_empty_preview_does_not_consume_parent_page_wheel(app):
    page = QScrollArea()
    page.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    preview = UploadPreview("Video", [".mp4"], "Videos (*.mp4)")
    layout.addWidget(preview)
    filler = QWidget()
    filler.setFixedHeight(600)
    layout.addWidget(filler)
    page.setWidget(content)
    page.resize(420, 240)
    page.show()
    _settle(app)
    try:
        assert preview.viewer.isHidden()
        assert page.verticalScrollBar().maximum() > 0
        # Dispatch through the window so Qt performs normal widget targeting
        # and propagation, rather than sending directly to a child label.
        point = preview.placeholder.mapTo(page, preview.placeholder.rect().center())
        QTest.wheelEvent(page.windowHandle(), point, QPoint(0, -120))
        _settle(app)
        assert page.verticalScrollBar().value() > 0
        _assert_empty(preview.viewer)
    finally:
        page.close()
