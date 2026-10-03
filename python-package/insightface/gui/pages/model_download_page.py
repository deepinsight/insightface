"""Manual GitHub release model downloads."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..app import (
    begin_context_activity,
    context_activity_count,
    end_context_activity,
)
from ..core.config import save_config
from ..core.constants import LICENSE_NOTICE
from ..core.i18n import tr
from ..core.model_downloads import (
    GITHUB_RELEASES_URL,
    ModelAsset,
    download_model_asset,
    installed_model_asset_path,
    is_model_asset_installed,
    load_cached_assets,
    local_model_status,
    refresh_model_assets,
)
from ..core.model_packages import (
    PERSON_MODEL_PACKAGES,
    inspect_person_model,
    is_gui_model_package_asset,
)
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage


class ModelDownloadPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Model Downloads",
            "Manually refresh GitHub release model URLs and download selected model packages locally.",
            parent,
        )
        self.assets: list[ModelAsset] = []
        self._download_in_progress = False
        self.content.addWidget(
            self.notice(
                "Downloads here start only when requested. Person Analysis also tries "
                "to download a missing Cheetah package when you start processing. "
                "Model files may have different licenses from code; review usage before deployment."
            )
        )
        self.content.addWidget(self.notice(LICENSE_NOTICE))
        self.use_selected_button = self.button(
            "Use Selected Model",
            self.use_selected_model,
            enabled=False,
        )
        self.download_selected_button = self.button(
            "Download Selected",
            self.download_selected,
            enabled=False,
        )
        self.content.addWidget(
            self.row(
                self.button("Refresh Download URLs", self.refresh_urls),
                self.download_selected_button,
                self.use_selected_button,
                self.button("Open Model Folder", self.open_model_folder),
                self.button("Open GitHub Releases", self.open_releases),
            )
        )
        self.table = QTableWidget(0, 7)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(
            [
                "asset",
                "source",
                "kind",
                "size",
                "updated_at",
                "local status",
                "download url",
            ]
        )
        configure_table_columns(self.table, [210, 100, 150, 90, 170, 150, 360])
        self.table.itemSelectionChanged.connect(self._update_selection_actions)
        # The surrounding tab can be shorter than the page's size hint. Let
        # the scrollable table shrink instead of overlapping the footer.
        self.content.addWidget(self.table, 1)
        self.content.addSpacing(8)
        self.source_footer = QFrame()
        self.source_footer.setObjectName("downloadSourceFooter")
        self.source_footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        footer_layout = QVBoxLayout(self.source_footer)
        footer_layout.setContentsMargins(10, 8, 10, 8)
        self.url_label = QLabel()
        self.url_label.setWordWrap(True)
        self.url_label.setProperty("role", "muted")
        footer_layout.addWidget(self.url_label)
        self.content.addWidget(self.source_footer)
        self.refresh()

    def refresh(self) -> None:
        self.assets = load_cached_assets(self.context.config.cache_dir)
        self.populate()

    def populate(self) -> None:
        self.table.setRowCount(len(self.assets))
        for row, asset in enumerate(self.assets):
            values = [
                asset.name,
                asset.source,
                asset.kind,
                self._format_size(asset.size),
                asset.updated_at,
                local_model_status(asset, self.context.config.model_root),
                asset.browser_download_url,
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        refresh_table_columns(self.table)
        self._update_selection_actions()
        language = self.context.config.ui_language
        self.url_label.setText(
            f"{tr('Refresh source', language)}: {GITHUB_RELEASES_URL}\n"
            f"{tr('Local model root', language)}: {Path(self.context.config.model_root).expanduser() / 'models'}"
        )

    def selected_asset(self) -> ModelAsset | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.assets):
            self.show_error("Select a model asset first.")
            return None
        return self.assets[row]

    def refresh_urls(self) -> None:
        def task():
            return refresh_model_assets(self.context.config.cache_dir)

        def done(payload):
            self.assets, message = payload
            self.populate()
            self.set_status(message)

        self.run_task("Refreshing model download URLs", task, done)

    def download_selected(self) -> None:
        if context_activity_count(
            self.context, "privateframe_jobs_in_progress"
        ) or context_activity_count(
            self.context, "person_analysis_jobs_in_progress"
        ):
            self.show_error(
                "Wait for analysis processing to finish before downloading "
                "a model package."
            )
            return
        if context_activity_count(
            self.context, "model_downloads_in_progress"
        ):
            self.show_error("A model download is already in progress.")
            return
        asset = self.selected_asset()
        if asset is None:
            return
        if is_model_asset_installed(asset, self.context.config.model_root):
            self.show_error(
                "The selected model asset is already downloaded and ready to use."
            )
            self._update_selection_actions()
            return

        self._download_in_progress = True
        begin_context_activity(self.context, "model_downloads_in_progress")
        self._update_selection_actions()
        manager = self.window()
        if hasattr(manager, "notify_model_files_changed"):
            manager.notify_model_files_changed()

        def task(progress=None, is_cancelled=None):
            del is_cancelled
            if progress:
                progress(0, asset.size or 1, f"Connecting to download {asset.name}")
            return download_model_asset(
                asset,
                model_root=self.context.config.model_root,
                gui_cache_dir=self.context.config.cache_dir,
                progress=progress,
            )

        def done(path):
            path = Path(path)
            lower_name = asset.name.lower()
            if "gfpgan" in lower_name:
                self.context.config.gfpgan_model_path = str(path)
                save_config(self.context.config)
            elif "swap" in lower_name or "inswapper" in lower_name:
                self.context.config.swap_model_path = str(path)
                save_config(self.context.config)
            self.populate()
            manager = self.window()
            if hasattr(manager, "refresh_model_pages"):
                manager.refresh_model_pages()
            self.set_status(f"Downloaded {asset.name} to {path}")

        def finished():
            self._download_in_progress = False
            end_context_activity(self.context, "model_downloads_in_progress")
            self._update_selection_actions()
            current_manager = self.window()
            if hasattr(current_manager, "notify_model_files_changed"):
                current_manager.notify_model_files_changed()

        try:
            self.run_task(
                f"Downloading {asset.name}",
                task,
                done,
                on_finished=finished,
            )
        except Exception:
            finished()
            raise

    def use_selected_model(self) -> None:
        asset = self.selected_asset()
        if asset is None:
            return
        if context_activity_count(
            self.context, "model_downloads_in_progress"
        ) or context_activity_count(
            self.context, "privateframe_jobs_in_progress"
        ) or context_activity_count(
            self.context, "person_analysis_jobs_in_progress"
        ):
            self.show_error(
                "Wait for active model or analysis work to finish before "
                "changing the global model."
            )
            return
        if not is_model_asset_installed(asset, self.context.config.model_root):
            self.show_error("Download the selected model before using it.")
            self._update_selection_actions()
            return
        use_kind = self._asset_use_kind(asset)
        if not use_kind:
            self.show_error(
                "This downloaded auxiliary model has no manual Use action."
            )
            self._update_selection_actions()
            return
        if use_kind == "global":
            self.context.config.model_name = asset.stem
            self.context.config.custom_model_dir = ""
            save_config(self.context.config)
            manager = self.window()
            if hasattr(manager, "notify_model_configuration_changed"):
                manager.notify_model_configuration_changed()
            if hasattr(manager, "refresh_model_pages"):
                manager.refresh_model_pages()
            if asset.stem in PERSON_MODEL_PACKAGES:
                self.set_status(f"Model set to {asset.stem}. Open Person Analysis to use it.")
            else:
                self.set_status(
                    f"Model set to {asset.stem}. Open Models and test model load."
                )
        else:
            path = installed_model_asset_path(
                asset,
                self.context.config.model_root,
            )
            if path is None:
                self.show_error("Download the selected model before using it.")
                return
            if use_kind == "gfpgan":
                self.context.config.gfpgan_model_path = str(path)
                save_config(self.context.config)
                self.set_status(f"GFPGAN model set to {path}. Enable GFPGAN in Models > Runtime to use it after face swap.")
            elif use_kind == "swap":
                self.context.config.swap_model_path = str(path)
                save_config(self.context.config)
                self.set_status(f"Face swap model set to {path}.")

    @staticmethod
    def _asset_use_kind(asset: ModelAsset | None) -> str:
        if asset is None:
            return ""
        if is_gui_model_package_asset(name=asset.name, source=asset.source):
            return "global"
        name = asset.name.casefold()
        if not name.endswith(".onnx"):
            return ""
        if "gfpgan" in name:
            return "gfpgan"
        if "swap" in name or "inswapper" in name:
            return "swap"
        return ""

    def _update_selection_actions(self) -> None:
        row = self.table.currentRow()
        asset = self.assets[row] if 0 <= row < len(self.assets) else None
        installed = bool(
            asset is not None
            and is_model_asset_installed(asset, self.context.config.model_root)
        )
        use_kind = self._asset_use_kind(asset)
        model_download_running = context_activity_count(
            self.context, "model_downloads_in_progress"
        )
        privateframe_running = context_activity_count(
            self.context, "privateframe_jobs_in_progress"
        )
        person_running = context_activity_count(
            self.context, "person_analysis_jobs_in_progress"
        )
        busy = bool(
            self._download_in_progress
            or model_download_running
            or privateframe_running
            or person_running
        )
        can_use = bool(use_kind and installed and not busy)
        can_download = bool(asset is not None and not installed and not busy)
        person_status = (
            inspect_person_model(asset.stem, self.context.config.model_root)
            if asset is not None and asset.stem in PERSON_MODEL_PACKAGES else None
        )
        if person_status is not None and person_status.state == "invalid":
            can_download = False
        self.use_selected_button.setEnabled(can_use)
        if busy:
            use_tooltip = (
                "Wait for active model or analysis work to finish before "
                "changing the global model."
            )
        elif asset is None:
            use_tooltip = "Select a downloaded model first."
        elif not installed:
            use_tooltip = "Download this model before using it."
        elif not use_kind:
            use_tooltip = (
                "This downloaded auxiliary model has no manual Use action."
            )
        else:
            use_tooltip = "Use this downloaded model."
        language = self.context.config.ui_language
        self.use_selected_button.setProperty(
            "_insightface_i18n_source_tooltip",
            use_tooltip,
        )
        self.use_selected_button.setToolTip(tr(use_tooltip, language))
        self.download_selected_button.setEnabled(
            can_download
        )
        if privateframe_running or person_running:
            download_tooltip = (
                "Wait for analysis processing to finish before downloading "
                "models."
            )
        elif model_download_running:
            download_tooltip = "A model download is already in progress."
        elif installed:
            download_tooltip = "This model asset is already downloaded."
        elif person_status is not None and person_status.state == "invalid":
            download_tooltip = person_status.message
        elif asset is None:
            download_tooltip = "Select a model asset to download."
        else:
            download_tooltip = "Download the selected asset."
        self.download_selected_button.setProperty(
            "_insightface_i18n_source_tooltip",
            download_tooltip,
        )
        self.download_selected_button.setToolTip(tr(download_tooltip, language))

    def open_model_folder(self) -> None:
        folder = Path(self.context.config.model_root).expanduser() / "models"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def open_releases(self) -> None:
        QDesktopServices.openUrl(QUrl(GITHUB_RELEASES_URL))

    @staticmethod
    def _format_size(size: int) -> str:
        if not size:
            return ""
        return f"{size / (1024 * 1024):.1f} MB"
