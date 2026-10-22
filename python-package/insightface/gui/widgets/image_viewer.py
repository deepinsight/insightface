"""Zoomable image viewer with face overlays."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        arr = np.ascontiguousarray(image)
        return QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(image[..., ::-1])
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()


class ImageViewer(QGraphicsView):
    faceClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageViewer")
        self.viewport().setObjectName("imageViewerViewport")
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.image: Optional[np.ndarray] = None
        self.faces: List[dict[str, Any]] = []
        self._fit_on_resize = False
        self._fitting_to_window = False

    def set_image(self, image: Optional[np.ndarray]) -> None:
        self.scene.clear()
        self.image = image
        self.faces = []
        self.pixmap_item = None
        self._fit_on_resize = False
        self.resetTransform()
        # clear() removes items, but the explicit rectangle from the previous
        # pixmap survives. Reset both scene and view overrides to the now-empty
        # scene; otherwise its old extent continues to create scroll bars.
        self.scene.setSceneRect(QRectF())
        self.setSceneRect(QRectF())
        if image is None:
            self.setDragMode(QGraphicsView.NoDrag)
            return
        pixmap = QPixmap.fromImage(numpy_to_qimage(image))
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.draw_overlays()
        self.fit_to_window()

    def set_faces(self, faces: Iterable[Any]) -> None:
        self.faces = []
        for face in faces:
            if hasattr(face, "bbox"):
                label = getattr(face, "label", None) or getattr(face, "person_name", None) or "Face"
                similarity = getattr(face, "similarity", None)
                self.faces.append({"bbox": face.bbox, "label": label, "similarity": similarity})
            else:
                self.faces.append(dict(face))
        self.draw_overlays()

    def draw_overlays(self) -> None:
        if self.pixmap_item is None or self.image is None:
            return
        for item in list(self.scene.items()):
            if item is not self.pixmap_item:
                self.scene.removeItem(item)
        for index, face in enumerate(self.faces):
            x1, y1, x2, y2 = [float(value) for value in face.get("bbox", [0, 0, 0, 0])]
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            pen = QPen(QColor("#10b981") if face.get("label") != "Unknown" else QColor("#f59e0b"))
            pen.setWidth(3)
            item = self.scene.addRect(rect, pen)
            item.setData(0, index)
            label = face.get("label", "Face")
            similarity = face.get("similarity")
            if similarity is not None:
                label = f"{label} {float(similarity):.2f}"
            text = self.scene.addText(label)
            text.setDefaultTextColor(QColor("#111827"))
            text.setPos(QPointF(x1, max(0, y1 - 24)))
            text.setData(0, index)

    def fit_to_window(self) -> None:
        if self.pixmap_item is None or self._fitting_to_window:
            return
        self._fit_on_resize = True
        self._fitting_to_window = True
        try:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        finally:
            self._fitting_to_window = False

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if getattr(self, "_fit_on_resize", False):
            self.fit_to_window()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if self.pixmap_item is None or not delta:
            event.ignore()
            return
        self._fit_on_resize = False
        factor = 1.2 if delta > 0 else 1 / 1.2
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        scene_pos = self.mapToScene(event.pos())
        for index, face in enumerate(self.faces):
            x1, y1, x2, y2 = [float(value) for value in face.get("bbox", [0, 0, 0, 0])]
            if x1 <= scene_pos.x() <= x2 and y1 <= scene_pos.y() <= y2:
                self.faceClicked.emit(index)
                break
        super().mousePressEvent(event)

    def render_with_overlay(self) -> Optional[np.ndarray]:
        if self.image is None:
            return None
        image = self.image.copy()
        try:
            import cv2

            for face in self.faces:
                x1, y1, x2, y2 = [int(round(float(value))) for value in face.get("bbox", [0, 0, 0, 0])]
                color = (16, 185, 129) if face.get("label") != "Unknown" else (245, 158, 11)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = face.get("label", "Face")
                if face.get("similarity") is not None:
                    label = f"{label} {float(face['similarity']):.2f}"
                cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        except Exception:
            pass
        return image
