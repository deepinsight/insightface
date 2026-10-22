"""Clickable drag-and-drop preview input for local images and videos."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
)

from ..core.tooltips import set_button_tooltip
from ..core.i18n import tr
from .image_viewer import ImageViewer


class UploadPreview(QFrame):
    """Preview box that doubles as a local file picker and drop target."""

    pathChanged = Signal(str)
    removed = Signal()

    def __init__(
        self,
        title: str,
        extensions: Iterable[str],
        dialog_filter: str,
        prompt: str = "Click to upload or drag a file here",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("uploadPreview")
        self.title = title
        self.prompt = prompt
        self.extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
        self.dialog_filter = dialog_filter
        self._path = ""
        self.setAcceptDrops(True)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setProperty("hoverActive", False)
        self.setProperty("dragActive", False)
        self.setProperty("hasFile", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedLayout()
        self.stack.setStackingMode(QStackedLayout.StackOne)
        self.viewer = ImageViewer()
        self.viewer.setAcceptDrops(True)
        self.viewer.setMouseTracking(True)
        self.viewer.setFrameShape(QFrame.NoFrame)
        self.viewer.setStyleSheet("background:transparent; border:0;")
        self.viewer.viewport().setAcceptDrops(True)
        self.viewer.viewport().setMouseTracking(True)
        self.viewer.viewport().setStyleSheet("background:transparent;")
        self.placeholder = QLabel(self._placeholder_text())
        self.placeholder.setObjectName("uploadPrompt")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.stack.addWidget(self.viewer)
        self.stack.addWidget(self.placeholder)
        self.stack.setCurrentWidget(self.placeholder)
        layout.addLayout(self.stack, 1)

        self.file_label = QLabel("")
        self.file_label.setObjectName("pathLabel")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.hide()
        layout.addWidget(self.file_label)

        self.remove_button = QPushButton("×", self)
        self.remove_button.setObjectName("removeUpload")
        set_button_tooltip(self.remove_button)
        self.remove_button.setFixedSize(22, 22)
        # clicked(bool) carries the checked state, not clear()'s emit option.
        self.remove_button.clicked.connect(lambda _checked=False: self.clear())
        self.remove_button.hide()

        for watched in (self, self.viewer, self.viewer.viewport(), self.placeholder):
            watched.installEventFilter(self)

    def path(self) -> str:
        return self._path

    def set_path(self, path: str, emit: bool = True) -> None:
        if not self._accepts_path(path):
            return
        self._path = str(Path(path).expanduser())
        # A newly selected file has no thumbnail yet. Do not leave the old
        # preview (including its zoom and scroll bars) beneath the prompt.
        self.viewer.set_image(None)
        self._show_selected()
        if emit:
            self.pathChanged.emit(self._path)

    def set_image(self, image: Optional[np.ndarray], path: str | None = None) -> None:
        if path:
            self._path = str(Path(path).expanduser())
        if image is None:
            self.viewer.set_image(None)
            if self._path:
                self._show_selected()
            else:
                self.clear(emit=False)
        else:
            self._show_selected(has_preview=True)
            self.viewer.set_image(image)

    def set_faces(self, faces) -> None:
        self.viewer.set_faces(faces)

    def render_with_overlay(self):
        return self.viewer.render_with_overlay()

    def clear(self, emit: bool = True) -> None:
        self._path = ""
        self.viewer.set_image(None)
        self.viewer.set_faces([])
        self.placeholder.setText(self._placeholder_text())
        self.stack.setCurrentWidget(self.placeholder)
        self.file_label.clear()
        self.file_label.hide()
        self.remove_button.hide()
        self._set_property("hasFile", False)
        if emit:
            self.removed.emit()
            self.pathChanged.emit("")

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"{tr('Select', self._language())} {tr(self.title, self._language())}",
            str(Path.home()),
            self.dialog_filter,
        )
        if path:
            self.set_path(path)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.DragEnter, QEvent.DragMove):
            if event.mimeData().hasUrls() and self._accepted_urls(event.mimeData().urls()):
                self._set_property("dragActive", True)
                event.acceptProposedAction()
                return True
            event.ignore()
            return True
        if event.type() == QEvent.DragLeave:
            self._set_property("dragActive", False)
            return False
        if event.type() == QEvent.Enter:
            self._set_property("hoverActive", True)
            return False
        if event.type() == QEvent.Leave:
            self._update_hover_from_cursor()
            return False
        if event.type() == QEvent.Drop:
            self._set_property("dragActive", False)
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            accepted = [path for path in paths if self._accepts_path(path)]
            if accepted:
                self.set_path(accepted[0])
                event.acceptProposedAction()
                return True
            event.ignore()
            return True
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and not self._path:
            self.browse()
            return True
        return super().eventFilter(watched, event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and not self._path:
            self.browse()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._set_property("hoverActive", True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._update_hover_from_cursor()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.remove_button.move(max(0, self.width() - 30), 8)
        self.remove_button.raise_()

    def _show_selected(self, has_preview: bool = False) -> None:
        self._set_property("hasFile", True)
        self.remove_button.show()
        if has_preview:
            self.stack.setCurrentWidget(self.viewer)
        else:
            name = Path(self._path).name if self._path else self.title
            self.placeholder.setText(
                f"{tr(self.title, self._language())}\n{name}\n{tr('Drop another file to replace it', self._language())}"
            )
            self.stack.setCurrentWidget(self.placeholder)
        if self._path:
            self.file_label.setText(str(self._path))
            self.file_label.show()
        # Switching a stacked page raises it above sibling widgets. Keep the
        # floating remove button above the newly visible preview or prompt.
        self.remove_button.raise_()

    def _placeholder_text(self) -> str:
        language = self._language()
        return f"{tr(self.title, language)}\n{tr(self.prompt, language)}"

    def _language(self) -> str | None:
        context = getattr(self.window(), "context", None)
        config = getattr(context, "config", None)
        return getattr(config, "ui_language", None)

    def _accepted_urls(self, urls) -> bool:
        return any(url.isLocalFile() and self._accepts_path(url.toLocalFile()) for url in urls)

    def _accepts_path(self, path: str) -> bool:
        if not path:
            return False
        p = Path(path).expanduser()
        return p.is_file() and (not self.extensions or p.suffix.lower() in self.extensions)

    def _set_property(self, name: str, value) -> None:
        self.setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def _update_hover_from_cursor(self) -> None:
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        self._set_property("hoverActive", inside)
