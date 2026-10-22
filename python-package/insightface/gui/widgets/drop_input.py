"""Reusable drag-and-drop file/folder input."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..core.tooltips import set_button_tooltip


class DropInput(QFrame):
    pathsChanged = Signal(list)
    removed = Signal()

    def __init__(
        self,
        title: str,
        mode: str = "file",
        extensions: Iterable[str] | None = None,
        dialog_filter: str = "All Files (*)",
        show_select_button: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.title = title
        self.mode = mode
        self.show_select_button = bool(show_select_button)
        self.extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or [])}
        self.dialog_filter = dialog_filter
        self._paths: list[str] = []
        self.setObjectName("dropInput")
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setProperty("hoverActive", False)
        self.setProperty("dragActive", False)
        self.setProperty("hasFiles", False)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: 600;")
        self.path_label = QLabel(self._empty_text())
        self.path_label.setObjectName("dropPrompt")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setWordWrap(True)
        self.path_label.setMinimumHeight(72)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        buttons = QHBoxLayout()
        self.select_button = QPushButton("Select")
        self.remove_button = QPushButton("Remove")
        self.remove_button.setEnabled(False)
        self.remove_button.setVisible(False)
        self.select_button.setVisible(self.show_select_button)
        self.select_button.clicked.connect(self.browse)
        self.remove_button.clicked.connect(self.clear)
        set_button_tooltip(self.select_button)
        set_button_tooltip(self.remove_button)
        buttons.addStretch(1)
        buttons.addWidget(self.select_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.path_label)
        layout.addLayout(buttons)
        for watched in (self, self.title_label, self.path_label, self.select_button, self.remove_button):
            watched.setAcceptDrops(True)
            watched.setMouseTracking(True)
            watched.installEventFilter(self)

    def paths(self) -> list[str]:
        return list(self._paths)

    def path(self) -> str:
        return self._paths[0] if self._paths else ""

    def set_path(self, path: str, emit: bool = True) -> None:
        self.set_paths([path] if path else [], emit=emit)

    def set_paths(self, paths: Sequence[str], emit: bool = True) -> None:
        accepted = [str(Path(path).expanduser()) for path in paths if self._accepts_path(path)]
        if self.mode != "files":
            accepted = accepted[:1]
        self._paths = accepted
        if accepted:
            display_text = self._selected_text(accepted)
            self.path_label.setText(display_text)
            self.path_label.setToolTip("\n".join(accepted))
            self.select_button.setVisible(False)
            self.remove_button.setEnabled(True)
            self.remove_button.setVisible(True)
        else:
            self.path_label.setText(self._empty_text())
            self.path_label.setToolTip("")
            self.select_button.setVisible(self.show_select_button)
            self.remove_button.setEnabled(False)
            self.remove_button.setVisible(False)
        if emit:
            self.pathsChanged.emit(self.paths())
        self._set_property("hasFiles", bool(accepted))

    def clear(self) -> None:
        self._paths = []
        self.path_label.setText(self._empty_text())
        self.path_label.setToolTip("")
        self.select_button.setVisible(self.show_select_button)
        self.remove_button.setEnabled(False)
        self.remove_button.setVisible(False)
        self._set_property("hasFiles", False)
        self.removed.emit()
        self.pathsChanged.emit([])

    def browse(self) -> None:
        if self.mode == "folder":
            path = QFileDialog.getExistingDirectory(self, f"Select {self.title}", str(Path.home()))
            if path:
                self.set_path(path)
        elif self.mode == "files":
            paths, _ = QFileDialog.getOpenFileNames(self, f"Select {self.title}", str(Path.home()), self.dialog_filter)
            if paths:
                self.set_paths(paths)
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"Select {self.title}", str(Path.home()), self.dialog_filter)
            if path:
                self.set_path(path)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and self._accepted_urls(event.mimeData().urls()):
            self._set_property("dragActive", True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and self._accepted_urls(event.mimeData().urls()):
            self._set_property("dragActive", True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_property("dragActive", False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self._set_property("dragActive", False)
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        accepted = [path for path in paths if self._accepts_path(path)]
        if accepted:
            self.set_paths(accepted)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _accepted_urls(self, urls) -> bool:
        return any(url.isLocalFile() and self._accepts_path(url.toLocalFile()) for url in urls)

    def _accepts_path(self, path: str) -> bool:
        if not path:
            return False
        p = Path(path)
        if self.mode == "folder":
            return p.is_dir()
        if p.is_dir():
            return self.mode == "folder"
        if self.extensions:
            return p.suffix.lower() in self.extensions
        return p.exists()

    def _empty_text(self) -> str:
        if self.mode == "folder":
            return "Click to upload or drag a folder here."
        if self.mode == "files":
            return "Click to upload or drag files here."
        return "Click to upload or drag a file here."

    def _selected_text(self, paths: Sequence[str]) -> str:
        prefix = "Selected folder:" if self.mode == "folder" else "Selected files:" if self.mode == "files" else "Selected file:"
        body = "\n".join(paths[:8])
        if len(paths) > 8:
            body = f"{body}\n... {len(paths)} total"
        return f"{prefix}\n{body}"

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
        if event.type() == QEvent.Drop:
            self._set_property("dragActive", False)
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            accepted = [path for path in paths if self._accepts_path(path)]
            if accepted:
                self.set_paths(accepted)
                event.acceptProposedAction()
                return True
            event.ignore()
            return True
        if event.type() == QEvent.Enter:
            self._set_property("hoverActive", True)
            return False
        if event.type() == QEvent.Leave:
            self._update_hover_from_cursor()
            return False
        if (
            event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
            and watched in {self, self.title_label, self.path_label}
        ):
            self.browse()
            return True
        return super().eventFilter(watched, event)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._set_property("hoverActive", True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._update_hover_from_cursor()
        super().leaveEvent(event)

    def _set_property(self, name: str, value) -> None:
        self.setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def _update_hover_from_cursor(self) -> None:
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        self._set_property("hoverActive", inside)
