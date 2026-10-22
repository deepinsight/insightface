"""Model runtime and download manager dialog."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QLabel, QTabWidget, QVBoxLayout

from ..core.i18n import tr
from ..pages.model_download_page import ModelDownloadPage
from ..pages.model_settings_page import ModelSettingsPage


class ModelManagerDialog(QDialog):
    # Configuration changes invalidate the ordinary GUI engine. Model-file
    # changes only refresh readiness displays (for example after a download).
    modelChanged = Signal()
    modelFilesChanged = Signal()

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("Models")
        self.setMinimumSize(1120, 700)
        self.resize(1240, 740)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.runtime_page = ModelSettingsPage(context)
        self.downloads_page = ModelDownloadPage(context)
        self.tabs.addTab(self.runtime_page, "Runtime")
        self.tabs.addTab(self.downloads_page, "Downloads")
        self.tabs.currentChanged.connect(self._refresh_current_tab)
        layout.addWidget(self.tabs, 1)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def open_page(self, label: str) -> None:
        if label == "Model Downloads":
            self.tabs.setCurrentWidget(self.downloads_page)
            self.downloads_page.refresh()
        elif label == "Model Settings":
            self.tabs.setCurrentWidget(self.runtime_page)
            self.runtime_page.refresh()

    def set_status(self, message: str) -> None:
        self.status_label.setText(tr(message, self.context.config.ui_language))
        if self.parent() is not None and hasattr(self.parent(), "set_status"):
            self.parent().set_status(message)

    def notify_model_configuration_changed(self) -> None:
        self.modelChanged.emit()

    def notify_model_files_changed(self) -> None:
        self.modelFilesChanged.emit()

    def run_task(
        self,
        title: str,
        fn,
        on_result=None,
        show_dialog: bool = True,
        on_progress=None,
        on_error=None,
        on_finished=None,
    ):
        parent = self.parent()
        if parent is not None and hasattr(parent, "run_task"):
            return parent.run_task(
                title,
                fn,
                on_result,
                show_dialog=show_dialog,
                on_progress=on_progress,
                on_error=on_error,
                on_finished=on_finished,
            )
        try:
            result = fn()
            if on_result:
                on_result(result)
        except Exception as exc:
            if on_error:
                on_error(str(exc))
            else:
                self.set_status(str(exc))
        finally:
            if on_finished:
                on_finished()
        return None

    def refresh_model_pages(self) -> None:
        self.runtime_page.refresh()
        self.downloads_page.populate()
        self.notify_model_files_changed()

    def _refresh_current_tab(self, index: int | None = None) -> None:
        del index
        widget = self.tabs.currentWidget()
        if hasattr(widget, "refresh"):
            widget.refresh()

    def refresh_statusbar(self) -> None:
        if self.parent() is not None and hasattr(self.parent(), "refresh_statusbar"):
            self.parent().refresh_statusbar()
