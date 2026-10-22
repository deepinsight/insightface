"""Single-page album import, clustering, and review workflow."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.clustering import cluster_embeddings_dbscan
from ..core.i18n import tr
from ..core.recognition import cosine_similarity, normalize_embedding
from ..core.tooltips import set_button_tooltip
from ..core.utils import crop_bbox, encode_webp_thumbnail, list_images, read_image, timestamp_for_filename
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage

DEFAULT_ALBUM_MIN_FACE_SIZE = 80
DEFAULT_ALBUM_COSINE_THRESHOLD = 0.48


class AlbumDirectoryList(QListWidget):
    foldersChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("albumDirectoryList")
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setToolTip("Drag one or more album folders here, or use Add Folder.")
        self.setProperty("hoverActive", False)
        self.setProperty("dragActive", False)
        self.installEventFilter(self)
        self.viewport().installEventFilter(self)

    def add_folder(self, folder: str) -> None:
        folder = str(Path(folder).expanduser())
        if folder and Path(folder).is_dir() and folder not in self.folders():
            self.addItem(folder)
            self.foldersChanged.emit()

    def folders(self) -> list[str]:
        return [self.item(index).text() for index in range(self.count())]

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._has_folder_urls(event):
            self._drag(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._has_folder_urls(event):
            self._drag(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._drag(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self._drag(False)
        added = False
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_dir():
                self.add_folder(url.toLocalFile())
                added = True
        if added:
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drag(self, active: bool) -> None:
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)

    @staticmethod
    def _has_folder_urls(event) -> bool:
        return event.mimeData().hasUrls() and any(
            Path(url.toLocalFile()).is_dir()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Enter:
            self._set_property("hoverActive", True)
            return False
        if event.type() == QEvent.Leave:
            self._update_hover_from_cursor()
            return False
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


class AlbumPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Album",
            "Import or refresh local album folders, cluster detected faces, and review the photos in each person group.",
            parent,
        )
        self.clusters: list[dict] = []
        self.cluster_items: dict[int, list[dict]] = {}
        self._loaded_saved_state = False

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.content.addWidget(
            self.notice(
                "All album processing is local. Import / Refresh scans every selected folder, extracts features "
                "only for new images, then runs DBSCAN clustering over all indexed faces using the selected cosine threshold."
            )
        )
        self.folder_list = AlbumDirectoryList()
        self.folder_list.setMinimumHeight(90)
        self.folder_list.foldersChanged.connect(self._save_directories)
        controls_layout.addWidget(QLabel("Album directories"))
        controls_layout.addWidget(self.folder_list)
        hint = QLabel("Drag album folders into the directory list, or click Add Folder.")
        hint.setProperty("role", "muted")
        controls_layout.addWidget(hint)
        button_row = QHBoxLayout()
        self.import_button = self._button("Import / Refresh", self.import_refresh)
        self.rebuild_button = self._button("Rebuild All", self.rebuild_all)
        for button in [
            self._button("Add Folder", self.add_folder),
            self._button("Remove Selected", self.remove_selected),
            self._button("Clear", self.clear_directories),
            self.import_button,
            self.rebuild_button,
        ]:
            button_row.addWidget(button)
        button_row.addStretch(1)
        controls_layout.addLayout(button_row)

        threshold_help = QLabel(
            "Cosine threshold is a face similarity cutoff: higher values make clusters stricter. "
            "DBSCAN internally uses cosine distance = 1 - cosine threshold."
        )
        threshold_help.setObjectName("albumThresholdHelp")
        threshold_help.setProperty("role", "muted")
        threshold_help.setWordWrap(True)
        controls_layout.addWidget(threshold_help)

        settings_row = QHBoxLayout()
        self.cluster_threshold = QDoubleSpinBox()
        self.cluster_threshold.setRange(0.01, 0.99)
        self.cluster_threshold.setSingleStep(0.01)
        self.cluster_threshold.setDecimals(2)
        self.cluster_threshold.setValue(DEFAULT_ALBUM_COSINE_THRESHOLD)
        self.cluster_threshold.setToolTip(
            "Cosine similarity threshold. Higher values make clusters stricter; default is 0.48."
        )
        self.min_samples = QSpinBox()
        self.min_samples.setRange(1, 50)
        self.min_samples.setValue(2)
        self.min_samples.setToolTip("Minimum neighboring faces needed for a DBSCAN core point.")
        self.min_face_size = QSpinBox()
        self.min_face_size.setRange(1, 4096)
        self.min_face_size.setValue(DEFAULT_ALBUM_MIN_FACE_SIZE)
        self.min_face_size.setToolTip(
            "Faces whose bounding box width or height is smaller than this value are skipped."
        )
        self.algorithm_label = QLabel("Algorithm: DBSCAN")
        settings_row.addWidget(QLabel("Cosine threshold"))
        settings_row.addWidget(self.cluster_threshold)
        settings_row.addWidget(QLabel("Min samples"))
        settings_row.addWidget(self.min_samples)
        settings_row.addWidget(QLabel("Minimum face size"))
        settings_row.addWidget(self.min_face_size)
        settings_row.addWidget(self.algorithm_label)
        settings_row.addStretch(1)
        controls_layout.addLayout(settings_row)
        self.content.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.cluster_table = QTableWidget(0, 2)
        self.cluster_table.setIconSize(QSize(56, 56))
        self.cluster_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cluster_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cluster_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cluster_table.setHorizontalHeaderLabels(["Thumbnail", "Photos"])
        configure_table_columns(self.cluster_table, [120, 80])
        self.cluster_table.currentCellChanged.connect(self.cluster_selected)
        splitter.addWidget(self.cluster_table)

        self.photo_table = QTableWidget(0, 3)
        self.photo_table.setIconSize(QSize(96, 72))
        self.photo_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.photo_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.photo_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.photo_table.setHorizontalHeaderLabels(["Thumbnail", "File", "Faces in cluster"])
        configure_table_columns(self.photo_table, [120, 360, 130])
        self.photo_table.cellDoubleClicked.connect(self.open_photo)
        splitter.addWidget(self.photo_table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.content.addWidget(splitter, 1)

    def _button(self, text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        set_button_tooltip(button)
        return button

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            tr("Select album folder", self.context.config.ui_language),
            str(Path.home()),
        )
        if folder:
            self.folder_list.add_folder(folder)

    def remove_selected(self) -> None:
        for item in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(item))
        self._save_directories()

    def clear_directories(self) -> None:
        self.folder_list.clear()
        self._save_directories()
        self.set_status("Album directories cleared. Existing clustering results are still available.")

    def import_refresh(self) -> None:
        self._run_import_refresh(rebuild=False)

    def rebuild_all(self) -> None:
        folders = [Path(folder) for folder in self.folder_list.folders()]
        if not folders:
            reply = QMessageBox.question(
                self,
                tr("Rebuild All", self.context.config.ui_language),
                tr(
                    "No album directories are selected. Rebuild All will clear saved album clustering results. Continue?",
                    self.context.config.ui_language,
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.context.storage.clear_album_results(
                    model_name=self._current_model_name()
                )
                self.clusters = []
                self.cluster_items = {}
                self._populate_clusters()
                self.set_status("Saved album clustering results were cleared.")
            return
        reply = QMessageBox.question(
            self,
            tr("Rebuild All", self.context.config.ui_language),
            tr(
                "Rebuild All will reprocess every image in the selected album directories and replace saved clustering results. Continue?",
                self.context.config.ui_language,
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._run_import_refresh(rebuild=True)

    def _run_import_refresh(self, rebuild: bool = False) -> None:
        folders = [Path(folder) for folder in self.folder_list.folders()]
        if not folders:
            self.show_error("Add one or more album directories first.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        engine = self.context.engine
        model_name = str(getattr(engine, "model_name", "") or "").strip()
        if not model_name:
            self.show_error("The loaded model has no model name.")
            return
        storage = self.context.storage
        self._save_directories()
        all_paths = []
        for folder in folders:
            all_paths.extend(str(path) for path in list_images(folder, recursive=True))
        if not all_paths:
            self.show_error("No supported images found in the selected directories.")
            return
        cosine_threshold = float(self.cluster_threshold.value())
        min_samples = int(self.min_samples.value())
        min_face_size = int(self.min_face_size.value())
        existing = (
            set()
            if rebuild
            else storage.existing_media_paths(
                all_paths,
                model_name=model_name,
            )
        )
        new_paths = list(all_paths) if rebuild else [path for path in all_paths if path not in existing]

        def task(progress=None, is_cancelled=None):
            deleted = 0
            if rebuild:
                deleted = storage.delete_media_faces_by_paths(
                    all_paths,
                    model_name=model_name,
                )
            imported = 0
            faces_saved = 0
            for index, path in enumerate(new_paths):
                if is_cancelled and is_cancelled():
                    break
                image = read_image(path)
                if image is None:
                    if progress:
                        progress(index + 1, len(new_paths), f"Skipped unreadable image: {Path(path).name}")
                    continue
                media_id = storage.add_media_item(
                    path,
                    "image",
                    width=image.shape[1],
                    height=image.shape[0],
                    file_size=Path(path).stat().st_size if Path(path).exists() else None,
                    mtime=Path(path).stat().st_mtime if Path(path).exists() else None,
                    thumbnail=encode_webp_thumbnail(image, max_side=120, quality=35),
                    processed_at=timestamp_for_filename(),
                )
                for face in engine.detect_faces(image, source_path=path):
                    if face.normed_embedding is None:
                        continue
                    if self._face_box_size(face.bbox) < min_face_size:
                        continue
                    face_image = face.crop if face.crop is not None else crop_bbox(image, face.bbox)
                    storage.add_media_face(
                        media_id,
                        face.normed_embedding,
                        crop_path="",
                        thumbnail=encode_webp_thumbnail(face_image, max_side=80, quality=70),
                        bbox=face.bbox,
                        kps=face.kps,
                        det_score=face.det_score,
                        quality_score=face.quality_score,
                        status="unknown",
                        model_name=model_name,
                    )
                    faces_saved += 1
                storage.mark_media_item_processed(media_id, model_name)
                imported += 1
                if progress:
                    progress(
                        index + 1,
                        max(1, len(new_paths)),
                        f"Imported {imported} new images, saved {faces_saved} faces",
                    )
            faces = self._faces_for_folders(
                folders,
                min_face_size,
                model_name=model_name,
                storage=storage,
            )
            cluster_items: dict[int, list[dict]] = {}
            clusters, algorithm = self._cluster_faces(
                faces,
                cosine_threshold,
                min_samples,
                cluster_items=cluster_items,
            )
            storage.save_album_results(
                clusters,
                cluster_items,
                algorithm,
                cluster_threshold=cosine_threshold,
                min_samples=min_samples,
                min_face_size=min_face_size,
                model_name=model_name,
            )
            return {
                "deleted": deleted,
                "imported": imported,
                "faces_saved": faces_saved,
                "clusters": clusters,
                "cluster_items": cluster_items,
                "algorithm": algorithm,
                "model_name": model_name,
            }

        def done(result):
            if (
                self.context.engine is not engine
                or self._current_model_name() != model_name
            ):
                self._load_saved_results(self._current_model_name())
                self.set_status(
                    f"Album processing for {model_name} completed, but the active model changed; "
                    "the current model's results are shown."
                )
                return
            self.clusters = result["clusters"]
            self.cluster_items = result["cluster_items"]
            self.algorithm_label.setText(f"Algorithm: {result['algorithm']}")
            self._populate_clusters()
            if rebuild:
                self.set_status(
                    f"Rebuilt album from scratch: removed {result['deleted']} indexed image(s), "
                    f"processed {result['imported']} image(s), saved {result['faces_saved']} face(s), "
                    f"built {len(self.clusters)} cluster(s)."
                )
            else:
                self.set_status(
                    f"Imported {result['imported']} new image(s), saved {result['faces_saved']} face(s), "
                    f"built {len(self.clusters)} cluster(s)."
                )

        self.run_task("Rebuilding album" if rebuild else "Importing and clustering album", task, done)

    def _faces_for_folders(
        self,
        folders: list[Path],
        min_face_size: int,
        *,
        model_name: str,
        storage=None,
    ) -> list[dict]:
        roots = [folder.resolve() for folder in folders]
        faces = []
        album_storage = storage or self.context.storage
        for face in album_storage.list_media_faces(model_name=model_name):
            try:
                media_path = Path(face["media_path"]).resolve()
            except Exception:
                continue
            if any(media_path == root or root in media_path.parents for root in roots):
                if (
                    face.get("embedding") is not None
                    and self._face_box_size(face.get("bbox")) >= min_face_size
                ):
                    faces.append(face)
        return faces

    def _cluster_faces(
        self,
        faces: list[dict],
        cosine_threshold: float,
        min_samples: int,
        *,
        cluster_items: dict[int, list[dict]] | None = None,
    ) -> tuple[list[dict], str]:
        embeddings = [face["embedding"] for face in faces]
        distance_threshold = max(0.01, 1.0 - float(cosine_threshold))
        labels, algorithm = cluster_embeddings_dbscan(
            embeddings,
            distance_threshold=distance_threshold,
            min_samples=min_samples,
        )
        groups: dict[int, list[dict]] = defaultdict(list)
        next_noise = max(labels, default=-1) + 1
        for face, label in zip(faces, labels):
            if label < 0:
                label = next_noise
                next_noise += 1
            groups[int(label)].append(face)
        next_album_id = 1
        clusters = []
        result_items: dict[int, list[dict]] = {}
        for label, items in groups.items():
            vectors = [normalize_embedding(item["embedding"]) for item in items if item.get("embedding") is not None]
            vectors = [vector for vector in vectors if vector is not None]
            if not vectors:
                continue
            centroid = normalize_embedding(np.mean(np.vstack(vectors), axis=0))
            cluster_id = next_album_id
            next_album_id += 1
            name = f"Album Person {cluster_id}"
            representative = max(items, key=lambda item: cosine_similarity(centroid, item.get("embedding")))
            photos = sorted({item["media_path"] for item in items})
            cluster = {
                "id": cluster_id,
                "label": label,
                "name": name,
                "source": "album",
                "face_count": len(items),
                "photo_count": len(photos),
                "avg_quality": sum(float(item.get("quality_score") or 0.0) for item in items) / max(1, len(items)),
                "thumbnail_face_id": representative.get("id"),
                "thumbnail_path": representative.get("media_path"),
                "photos": photos,
            }
            clusters.append(cluster)
            result_items[cluster_id] = items
        if cluster_items is None:
            self.cluster_items = result_items
        else:
            cluster_items.update(result_items)
        return sorted(clusters, key=lambda row: (-row["face_count"], row["id"])), algorithm

    def _save_directories(self) -> None:
        self.context.storage.save_album_directories(self.folder_list.folders())

    def refresh(self) -> None:
        if not self._loaded_saved_state:
            self._loaded_saved_state = True
            self.folder_list.blockSignals(True)
            self.folder_list.clear()
            for folder in self.context.storage.list_album_directories():
                if Path(folder).expanduser().is_dir():
                    self.folder_list.addItem(folder)
            self.folder_list.blockSignals(False)
        self._load_saved_results(self._current_model_name())

    def _load_saved_results(self, model_name: str) -> None:
        data = self.context.storage.load_album_results(model_name=model_name)
        clusters = data.get("clusters") if isinstance(data, dict) else None
        if not isinstance(clusters, list):
            self.clusters = []
            self.cluster_items = {}
            self.algorithm_label.setText("Algorithm: DBSCAN")
            self._populate_clusters()
            return
        faces_by_id = {
            int(face["id"]): face
            for face in self.context.storage.list_media_faces(model_name=model_name)
            if face.get("id") is not None
        }
        self.clusters = []
        self.cluster_items = {}
        for cluster in clusters:
            try:
                cluster_id = int(cluster["id"])
            except Exception:
                continue
            face_ids = [int(face_id) for face_id in cluster.get("face_ids", []) if str(face_id).isdigit()]
            items = [
                faces_by_id[face_id]
                for face_id in face_ids
                if face_id in faces_by_id
            ]
            if not items:
                continue
            self.cluster_items[cluster_id] = items
            self.clusters.append(cluster)
        algorithm = data.get("algorithm")
        if algorithm not in {"DBSCAN", "centroid fallback", "none"}:
            algorithm = "DBSCAN"
        self.algorithm_label.setText(f"Algorithm: {algorithm}")
        if data.get("cluster_threshold") is not None:
            self.cluster_threshold.setValue(float(data["cluster_threshold"]))
        if data.get("min_samples") is not None:
            self.min_samples.setValue(int(data["min_samples"]))
        if data.get("min_face_size") is not None:
            self.min_face_size.setValue(int(data["min_face_size"]))
        self._populate_clusters()

    def _current_model_name(self) -> str:
        return str(getattr(self.context.engine, "model_name", "") or "").strip()

    def _populate_clusters(self) -> None:
        self.cluster_table.setRowCount(len(self.clusters))
        for row, cluster in enumerate(self.clusters):
            values = [
                "",
                cluster.get("photo_count", 0),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 0:
                    icon = self._cluster_thumbnail_icon(cluster, QSize(56, 56))
                    if icon:
                        item.setIcon(icon)
                item.setData(Qt.UserRole, cluster.get("id"))
                item.setTextAlignment(Qt.AlignCenter)
                item.setToolTip(
                    f"ID: {cluster.get('id', '')}\n"
                    f"Name: {cluster.get('name', '')}\n"
                    f"Faces: {cluster.get('face_count', 0)}\n"
                    f"Photos: {cluster.get('photo_count', 0)}"
                )
                self.cluster_table.setItem(row, col, item)
            self.cluster_table.setRowHeight(row, 64)
        refresh_table_columns(self.cluster_table)
        if self.clusters:
            self.cluster_table.selectRow(0)
            self._populate_photos(int(self.clusters[0].get("id", 0)))
        else:
            self.photo_table.setRowCount(0)

    def cluster_selected(self, current_row: int, current_column: int, previous_row: int, previous_column: int) -> None:
        del current_column, previous_row, previous_column
        if current_row < 0 or current_row >= len(self.clusters):
            self.photo_table.setRowCount(0)
            return
        self._populate_photos(int(self.clusters[current_row].get("id", 0)))

    def _populate_photos(self, cluster_id: int) -> None:
        items = self.cluster_items.get(cluster_id, [])
        grouped: dict[str, dict] = {}
        for item in items:
            path = item["media_path"]
            if path not in grouped:
                grouped[path] = {"count": 0, "thumbnail": item.get("media_thumbnail")}
            grouped[path]["count"] += 1
        if not grouped:
            for cluster in self.clusters:
                if int(cluster.get("id", -1)) == cluster_id:
                    for path in cluster.get("photos", []):
                        grouped[str(path)] = {"count": 1, "thumbnail": None}
                    break
        rows = sorted(grouped.items())
        self.photo_table.setRowCount(len(rows))
        for row, (path, data) in enumerate(rows):
            thumb = QTableWidgetItem("")
            icon = self._icon_from_bytes(data.get("thumbnail"), QSize(96, 72))
            if icon:
                thumb.setIcon(icon)
            thumb.setData(Qt.UserRole, path)
            file_item = QTableWidgetItem(path)
            file_item.setData(Qt.UserRole, path)
            self.photo_table.setItem(row, 0, thumb)
            self.photo_table.setItem(row, 1, file_item)
            self.photo_table.setItem(row, 2, QTableWidgetItem(str(data["count"])))
            self.photo_table.setRowHeight(row, 80)
        refresh_table_columns(self.photo_table)

    def open_photo(self, row: int, column: int) -> None:
        item = self.photo_table.item(row, 0) or self.photo_table.item(row, 1)
        if item is None:
            return
        path = item.data(Qt.UserRole)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _face_box_size(bbox) -> float:
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        except Exception:
            return 0.0
        return min(abs(x2 - x1), abs(y2 - y1))

    def _cluster_thumbnail_icon(self, cluster: dict, size: QSize) -> QIcon | None:
        thumbnail_face_id = cluster.get("thumbnail_face_id")
        if thumbnail_face_id is not None:
            for item in self.cluster_items.get(int(cluster.get("id", -1)), []):
                if item.get("id") == thumbnail_face_id:
                    return self._icon_from_bytes(item.get("thumbnail"), size)
        for item in self.cluster_items.get(int(cluster.get("id", -1)), []):
            icon = self._icon_from_bytes(item.get("thumbnail"), size)
            if icon is not None:
                return icon
        return None

    @staticmethod
    def _icon_from_bytes(data, size: QSize) -> QIcon | None:
        if not data:
            return None
        try:
            from PIL import Image

            with Image.open(BytesIO(bytes(data))) as image:
                rgba = image.convert("RGBA")
                raw = rgba.tobytes("raw", "RGBA")
                qimage = QImage(raw, rgba.width, rgba.height, QImage.Format_RGBA8888).copy()
            pixmap = QPixmap.fromImage(qimage)
        except Exception:
            return None
        return QIcon(pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
