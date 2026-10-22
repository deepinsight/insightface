"""Base page helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.constants import LOCAL_PROCESSING_NOTICE
from ..core.i18n import tr
from ..core.tooltips import set_button_tooltip


class BasePage(QWidget):
    def __init__(self, context, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.context = context
        self.title = title
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setSpacing(12)
        header = QLabel(title)
        header.setStyleSheet("font-size: 24px; font-weight: 700;")
        self.root_layout.addWidget(header)
        if description:
            desc = QLabel(description)
            desc.setWordWrap(True)
            desc.setProperty("role", "muted")
            self.root_layout.addWidget(desc)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("role", "status")
        self.content = QVBoxLayout()
        self.root_layout.addLayout(self.content, 1)
        self.root_layout.addWidget(self.status_label)

    def notice(self, text: str = LOCAL_PROCESSING_NOTICE) -> QLabel:
        label = QLabel(text)
        label.setObjectName("noticeLabel")
        label.setWordWrap(True)
        return label

    def card(self) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        return frame, layout

    def row(self, *widgets) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        return wrapper

    def button(self, text: str, callback: Callable, enabled: bool = True) -> QPushButton:
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        btn.setEnabled(enabled)
        set_button_tooltip(btn)
        return btn

    def choose_file(self, caption: str, filters: str = "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)") -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(self, caption, str(Path.home()), filters)
        return path or None

    def choose_files(self, caption: str, filters: str = "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)") -> list[str]:
        paths, _ = QFileDialog.getOpenFileNames(self, caption, str(Path.home()), filters)
        return paths

    def choose_folder(self, caption: str) -> Optional[str]:
        path = QFileDialog.getExistingDirectory(self, caption, str(Path.home()))
        return path or None

    def save_file(self, caption: str, default_name: str, filters: str) -> Optional[str]:
        path, _ = QFileDialog.getSaveFileName(self, caption, default_name, filters)
        return path or None

    def set_status(self, message: str) -> None:
        self.status_label.setText(tr(message, self.context.config.ui_language))
        main = self.window()
        if main is not self and hasattr(main, "set_status"):
            main.set_status(message)

    def show_error(self, message: str) -> None:
        self.set_status(message)
        QMessageBox.warning(self, tr(self.title, self.context.config.ui_language), tr(message, self.context.config.ui_language))

    def run_task(
        self,
        title: str,
        fn: Callable,
        on_result: Callable | None = None,
        show_dialog: bool = True,
        on_progress: Callable | None = None,
        on_error: Callable | None = None,
        on_finished: Callable | None = None,
    ):
        main = self.window()
        if hasattr(main, "run_task"):
            return main.run_task(
                title,
                fn,
                on_result,
                show_dialog=show_dialog,
                on_progress=on_progress,
                on_error=on_error,
                on_finished=on_finished,
            )
        else:
            try:
                result = fn()
                if on_result:
                    on_result(result)
            except Exception as exc:
                if on_error:
                    on_error(str(exc))
                else:
                    self.show_error(str(exc))
            finally:
                if on_finished:
                    on_finished()
        return None

    def refresh(self) -> None:
        pass
