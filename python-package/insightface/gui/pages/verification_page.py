"""Combined Query and Gallery verification workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QDir, QEvent, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileSystemModel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..core.config import save_config
from ..core.constants import DEFAULT_THRESHOLD, IMAGE_EXTENSIONS
from ..core.evaluation import (
    MULTI_FACE_REQUIRE_ONE,
    MULTI_FACE_SKIP,
    MULTI_FACE_USE_CENTERED_LARGEST,
    MULTI_FACE_USE_LARGEST,
    multi_face_policy_help,
    select_face_by_policy,
)
from ..core.i18n import apply_translations, tr
from ..core.recognition import compare_embeddings
from ..core.tooltips import apply_button_tooltips, set_button_tooltip
from ..core.utils import list_images, read_image
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from ..widgets.upload_preview import UploadPreview
from .base import BasePage


class GalleryUploadPanel(QFrame):
    """Clickable and draggable gallery input for image files or folders."""

    pathsChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[str] = []
        self.setObjectName("galleryUpload")
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.setProperty("hoverActive", False)
        self.setProperty("dragActive", False)
        self.setProperty("hasFiles", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        header = QHBoxLayout()
        title = QLabel("Gallery")
        title.setStyleSheet("font-size:15px; font-weight:700;")
        self.count_label = QLabel("No images")
        self.count_label.setProperty("role", "muted")
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        set_button_tooltip(self.remove_button)
        set_button_tooltip(self.clear_button)
        header.addWidget(title)
        header.addWidget(self.count_label)
        header.addStretch(1)
        header.addWidget(self.remove_button)
        header.addWidget(self.clear_button)
        layout.addLayout(header)

        self.prompt = QLabel("Click to upload images or a folder, or drag them here")
        self.prompt.setObjectName("uploadPrompt")
        self.prompt.setAlignment(Qt.AlignCenter)
        self.prompt.setWordWrap(True)
        self.prompt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.prompt, 1)

        self.single_preview = QLabel()
        self.single_preview.setAlignment(Qt.AlignCenter)
        self.single_preview.setMinimumHeight(180)
        self.single_preview.setStyleSheet("border:0; background:transparent;")
        self.single_preview.hide()
        self.single_hint = QLabel("")
        self.single_hint.setObjectName("pathLabel")
        self.single_hint.setAlignment(Qt.AlignCenter)
        self.single_hint.setWordWrap(True)
        self.single_hint.hide()
        layout.addWidget(self.single_preview, 1)
        layout.addWidget(self.single_hint)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(56, 56))
        self.list_widget.itemDoubleClicked.connect(self.open_item)
        layout.addWidget(self.list_widget, 1)
        self.list_widget.hide()
        self.remove_button.hide()
        self.clear_button.hide()

        for watched in (self, self.prompt, self.single_preview, self.single_hint, self.list_widget, self.list_widget.viewport()):
            watched.installEventFilter(self)

    def paths(self) -> list[str]:
        return list(self._paths)

    def set_paths(self, paths: list[str], emit: bool = True) -> None:
        resolved = self._expand_paths(paths)
        self._paths = resolved
        self._render()
        if emit:
            self.pathsChanged.emit(self.paths())

    def add_paths(self, paths: list[str]) -> None:
        self.set_paths(self._paths + paths)

    def clear(self) -> None:
        self.set_paths([])

    def remove_selected(self) -> None:
        remove = {item.data(Qt.UserRole) for item in self.list_widget.selectedItems()}
        if not remove:
            return
        self.set_paths([path for path in self._paths if path not in remove])

    def browse(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Select gallery images or folders")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        hint = QLabel("Select one or more image files or folders. Folders are scanned recursively.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        model = QFileSystemModel(dialog)
        model.setRootPath(str(Path.home()))
        model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot | QDir.Readable)
        model.setNameFilters(["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"])
        model.setNameFilterDisables(False)
        tree = QTreeView(dialog)
        tree.setModel(model)
        tree.setRootIndex(model.index(str(Path.home())))
        tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tree.setColumnWidth(0, 320)
        tree.doubleClicked.connect(lambda index: tree.setRootIndex(index) if model.isDir(index) else None)
        layout.addWidget(tree, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        apply_button_tooltips(dialog)
        apply_translations(dialog, self._language())
        if dialog.exec() == QDialog.Accepted:
            indexes = tree.selectionModel().selectedRows(0)
            paths = [model.filePath(index) for index in indexes]
            if paths:
                self.add_paths(paths)

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
            expanded = self._expand_paths(paths)
            if expanded:
                self.add_paths(expanded)
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
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and watched in {self.prompt, self.single_preview, self.single_hint}:
            self.browse()
            return True
        return super().eventFilter(watched, event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and not self._paths:
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

    def open_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _render(self) -> None:
        self.list_widget.clear()
        self.single_preview.clear()
        for path in self._paths:
            item = QListWidgetItem(Path(path).name)
            icon = self._icon(path, QSize(56, 56))
            if icon is not None:
                item.setIcon(icon)
            item.setToolTip(path)
            item.setData(Qt.UserRole, path)
            self.list_widget.addItem(item)
        has_files = bool(self._paths)
        single = len(self._paths) == 1
        self.prompt.setVisible(not has_files)
        self.single_preview.setVisible(single)
        self.single_hint.setVisible(single)
        self.list_widget.setVisible(has_files and not single)
        self.remove_button.setVisible(has_files and not single)
        self.clear_button.setVisible(has_files)
        count = len(self._paths)
        language = self._language()
        if count:
            image_label = "image" if count == 1 else "images"
            self.count_label.setText(f"{count} {tr(image_label, language)}")
        else:
            self.count_label.setText(tr("No images", language))
        if single:
            self._update_single_preview()
        self._set_property("hasFiles", has_files)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if len(self._paths) == 1:
            self._update_single_preview()

    def _expand_paths(self, paths: list[str]) -> list[str]:
        collected: list[str] = []
        seen = set()
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            candidates: list[Path]
            if path.is_dir():
                candidates = list_images(path, recursive=True)
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                candidates = [path]
            else:
                candidates = []
            for candidate in candidates:
                value = str(candidate)
                if value not in seen and value not in collected:
                    collected.append(value)
                    seen.add(value)
        return collected

    def _accepted_urls(self, urls) -> bool:
        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile()).expanduser()
            if path.is_dir() or (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                return True
        return False

    def _set_property(self, name: str, value) -> None:
        self.setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def _update_hover_from_cursor(self) -> None:
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        self._set_property("hoverActive", inside)

    @staticmethod
    def _icon(path: str | None, size: QSize) -> QIcon | None:
        if not path:
            return None
        pixmap = QPixmap(str(Path(path)))
        if pixmap.isNull():
            return None
        return QIcon(pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_single_preview(self) -> None:
        if len(self._paths) != 1:
            return
        path = self._paths[0]
        pixmap = QPixmap(str(Path(path)))
        if pixmap.isNull():
            self.single_preview.setText(Path(path).name)
        else:
            size = self.single_preview.size()
            target = QSize(max(180, size.width() - 24), max(150, size.height() - 12))
            self.single_preview.setPixmap(pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.single_hint.setText(
            f"{Path(path).name}\n{tr('Click or drag more images or folders to add to Gallery.', self._language())}"
        )

    def _language(self) -> str | None:
        context = getattr(self.window(), "context", None)
        config = getattr(context, "config", None)
        return getattr(config, "ui_language", None)


class VerificationPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Face Recognition",
            "Upload one query image and a gallery image, image set, or folder. One gallery image runs 1:1 compare; multiple gallery images run 1:N gallery search.",
            parent,
        )
        self.query_path = ""
        self.query_image: np.ndarray | None = None
        self.gallery_paths: list[str] = []
        self._gallery_embedding_cache_key: tuple[object, ...] | None = None
        self._gallery_embedding_cache: list[dict] | None = None
        self._active_engine_cache_identity = self._engine_cache_identity(
            context.engine
        )
        self.results: list[dict] = []
        if abs(float(context.config.recognition_threshold) - DEFAULT_THRESHOLD) > 1e-9:
            context.config.recognition_threshold = DEFAULT_THRESHOLD
            save_config(context.config)

        self.workflow_notice = self.notice("")
        self.content.addWidget(self.workflow_notice)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.query_input = UploadPreview(
            "Query",
            extensions=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            dialog_filter="Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)",
        )
        self.query_input.pathChanged.connect(lambda path: self.load_query(path) if path else self.clear_query())
        left_layout.addWidget(self.query_input)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_input = GalleryUploadPanel()
        self.gallery_input.pathsChanged.connect(self.set_gallery_paths)
        right_layout.addWidget(self.gallery_input)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self.content.addWidget(splitter, 1)

        control_row = QHBoxLayout()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 0.99)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(2)
        self.threshold.setValue(DEFAULT_THRESHOLD)
        self.threshold.setToolTip("Similarity threshold for matching query and gallery faces.")
        self.multi_face_policy = QComboBox()
        self.multi_face_policy.setProperty("i18nItems", True)
        self.multi_face_policy.addItem("Require exactly one face", MULTI_FACE_REQUIRE_ONE)
        self.multi_face_policy.addItem("Use largest face", MULTI_FACE_USE_LARGEST)
        self.multi_face_policy.addItem("Use largest centered face", MULTI_FACE_USE_CENTERED_LARGEST)
        self.multi_face_policy.addItem("Mark as skip", MULTI_FACE_SKIP)
        self.multi_face_policy.setObjectName("verificationMultiFacePolicy")
        self.multi_face_policy.setToolTip("Choose how query and gallery images handle multiple detected faces.")
        default_policy_index = self.multi_face_policy.findData(MULTI_FACE_USE_CENTERED_LARGEST)
        if default_policy_index >= 0:
            self.multi_face_policy.setCurrentIndex(default_policy_index)
        self.multi_face_policy.currentIndexChanged.connect(self._multi_face_policy_changed)
        self.run_button = QPushButton("Run Recognition")
        self.run_button.clicked.connect(self.run_verification)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_all)
        set_button_tooltip(self.run_button)
        set_button_tooltip(self.clear_button)
        control_row.addWidget(QLabel("Recognition threshold"))
        control_row.addWidget(self.threshold)
        control_row.addWidget(QLabel("Multi-face handling"))
        control_row.addWidget(self.multi_face_policy)
        control_row.addStretch(1)
        control_row.addWidget(self.run_button)
        control_row.addWidget(self.clear_button)
        self.content.addLayout(control_row)

        self.result_table = QTableWidget(0, 7)
        self.result_table.setIconSize(QSize(56, 56))
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setHorizontalHeaderLabels(
            ["rank", "thumbnail", "gallery_file", "similarity", "threshold", "decision", "det_score"]
        )
        configure_table_columns(self.result_table, [56, 86, 320, 92, 92, 130, 92])
        self.result_table.cellDoubleClicked.connect(self.open_result)
        self.content.addWidget(self.result_table, 1)
        self._update_policy_notice()

    def load_query(self, path: str) -> None:
        image = read_image(path)
        if image is None:
            self.query_input.clear(emit=False)
            self.show_error("Image read failure.")
            return
        self.query_path = path
        self.query_image = image
        self.query_input.set_image(image, path)
        self._update_mode_label()

    def clear_query(self) -> None:
        self.query_path = ""
        self.query_image = None
        self.query_input.clear(emit=False)
        self.query_input.set_faces([])
        self._update_mode_label()

    def set_gallery_paths(self, paths: list[str]) -> None:
        new_paths = list(paths)
        if tuple(new_paths) != tuple(self.gallery_paths):
            self._clear_gallery_embedding_cache()
        self.gallery_paths = new_paths
        self.results = []
        self.result_table.setRowCount(0)
        self._update_mode_label()

    def clear_all(self) -> None:
        self.clear_query()
        self.gallery_input.clear()
        self.results = []
        self.result_table.setRowCount(0)
        self.set_status("Cleared.")

    def run_verification(self) -> None:
        if self.query_image is None or not self.query_path:
            self.show_error("Please upload a query image.")
            return
        if not self.gallery_paths:
            self.show_error("Please upload one or more gallery images or folders.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return

        engine = self.context.engine
        engine_identity = self._engine_cache_identity(engine)
        query_image = self.query_image.copy()
        query_path = self.query_path
        gallery_paths = list(self.gallery_paths)
        multi_face_policy = self._multi_face_policy()
        gallery_key = (engine_identity, multi_face_policy, *gallery_paths)
        cached_gallery = (
            list(self._gallery_embedding_cache)
            if self._gallery_embedding_cache_key == gallery_key and self._gallery_embedding_cache is not None
            else None
        )
        threshold = self.threshold.value()

        def task(progress=None, is_cancelled=None):
            query_faces = engine.detect_faces(query_image, source_path=query_path)
            query_face = select_face_by_policy(
                query_faces,
                query_image.shape,
                multi_face_policy,
                path=query_path,
                stage="query",
            )
            if query_face is None or query_face.normed_embedding is None:
                raise ValueError("No face detected in the query image or embedding unavailable.")
            if cached_gallery is not None:
                gallery_cache = cached_gallery
                if progress:
                    progress(1, 1, "Using cached gallery embeddings")
            else:
                gallery_cache = self._build_gallery_embedding_cache(
                    engine,
                    gallery_paths,
                    multi_face_policy,
                    progress,
                    is_cancelled,
                )
            rows = self._compare_query_to_gallery_cache(query_face.normed_embedding, gallery_cache, threshold)
            rows.sort(
                key=lambda item: item["similarity"] if isinstance(item.get("similarity"), (int, float)) else -1.0,
                reverse=True,
            )
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank
            mode = "1:1 Compare" if len(gallery_paths) == 1 else "1:N Gallery Search"
            return {
                "query_bbox": query_face.bbox,
                "results": rows,
                "mode": mode,
                "query_path": query_path,
                "gallery_paths": tuple(gallery_paths),
                "gallery_key": gallery_key,
                "gallery_cache": gallery_cache,
                "multi_face_policy": multi_face_policy,
                "engine_identity": engine_identity,
            }

        def done(payload):
            if (
                self.context.engine is not engine
                or self._engine_cache_identity(self.context.engine)
                != payload["engine_identity"]
            ):
                self.set_status(
                    "Recognition result ignored because the global model changed "
                    "during processing."
                )
                return
            if (
                self.query_path != payload["query_path"]
                or tuple(self.gallery_paths) != payload["gallery_paths"]
                or self._multi_face_policy() != payload["multi_face_policy"]
            ):
                self.set_status("Recognition result ignored because query, gallery, or multi-face handling changed during processing.")
                return
            if cached_gallery is None:
                self._gallery_embedding_cache_key = payload["gallery_key"]
                self._gallery_embedding_cache = payload["gallery_cache"]
            self.results = payload["results"]
            self.query_input.set_faces([{"bbox": payload["query_bbox"], "label": "Query"}])
            self._populate_results()
            cache_status = "cached gallery embeddings" if cached_gallery is None else "reused cached gallery embeddings"
            self.set_status(f"{payload['mode']} complete with {cache_status}. {len(self.results)} gallery result(s).")

        self.run_task("Running verification", task, done)

    def open_result(self, row: int, column: int) -> None:
        del column
        if row < 0 or row >= len(self.results):
            return
        path = self.results[row].get("path")
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _populate_results(self) -> None:
        self.result_table.setRowCount(len(self.results))
        for row_index, result in enumerate(self.results):
            values = [
                result.get("rank", ""),
                "",
                Path(str(result.get("path", ""))).name if result.get("path") else "",
                f"{float(result.get('similarity', 0.0)):.4f}" if result.get("similarity") is not None else "",
                f"{float(result.get('threshold', 0.0)):.4f}" if result.get("threshold") is not None else "",
                result.get("decision", ""),
                f"{float(result.get('det_score', 0.0)):.4f}" if result.get("det_score") is not None else "",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 1:
                    icon = GalleryUploadPanel._icon(result.get("path"), QSize(56, 56))
                    if icon is not None:
                        item.setIcon(icon)
                item.setToolTip(str(result.get("notes") or result.get("path") or ""))
                item.setData(Qt.UserRole, result.get("path", ""))
                self.result_table.setItem(row_index, col, item)
            self.result_table.setRowHeight(row_index, 64)
        refresh_table_columns(self.result_table)

    def _update_mode_label(self) -> None:
        pass

    def _clear_gallery_embedding_cache(self) -> None:
        self._gallery_embedding_cache_key = None
        self._gallery_embedding_cache = None

    def _build_gallery_embedding_cache(
        self,
        engine,
        gallery_paths: list[str],
        multi_face_policy: str,
        progress=None,
        is_cancelled=None,
    ) -> list[dict]:
        cache: list[dict] = []
        for index, path in enumerate(gallery_paths):
            if is_cancelled and is_cancelled():
                break
            image = read_image(path)
            if image is None:
                cache.append(self._gallery_error_cache_item(path, "Image read failure."))
            else:
                try:
                    faces = engine.detect_faces(image, source_path=path)
                    face = select_face_by_policy(
                        faces,
                        image.shape,
                        multi_face_policy,
                        path=path,
                        stage="gallery",
                    )
                    if face is None or face.normed_embedding is None:
                        cache.append(self._gallery_error_cache_item(path, "No face detected."))
                    else:
                        cache.append(
                            {
                                "path": path,
                                "embedding": np.asarray(face.normed_embedding, dtype=np.float32).copy(),
                                "det_score": face.det_score,
                                "bbox": face.bbox,
                                "notes": "",
                            }
                        )
                except Exception as exc:
                    cache.append(self._gallery_error_cache_item(path, str(exc)))
            if progress:
                progress(
                    index + 1,
                    len(gallery_paths),
                    f"Computed gallery embeddings for {index + 1} of {len(gallery_paths)} images",
                )
        return cache

    def _compare_query_to_gallery_cache(
        self, query_embedding: np.ndarray, gallery_cache: list[dict], threshold: float
    ) -> list[dict]:
        rows = []
        for item in gallery_cache:
            embedding = item.get("embedding")
            if embedding is None:
                rows.append(
                    self._error_row(
                        str(item.get("path", "")),
                        str(item.get("notes") or "No face detected."),
                        threshold,
                    )
                )
                continue
            result = compare_embeddings(query_embedding, embedding, threshold)
            rows.append(
                {
                    "rank": 0,
                    "path": item.get("path", ""),
                    "similarity": float(result["similarity"]),
                    "threshold": threshold,
                    "decision": str(result["decision"]),
                    "det_score": item.get("det_score"),
                    "bbox": item.get("bbox"),
                    "notes": item.get("notes", ""),
                }
            )
        return rows

    @staticmethod
    def _gallery_error_cache_item(path: str, message: str) -> dict:
        return {
            "path": path,
            "embedding": None,
            "det_score": None,
            "bbox": None,
            "notes": message,
        }

    @staticmethod
    def _error_row(path: str, message: str, threshold: float) -> dict:
        return {
            "rank": 0,
            "path": path,
            "similarity": None,
            "threshold": threshold,
            "decision": message,
            "det_score": None,
            "bbox": None,
            "notes": message,
        }

    def _multi_face_policy(self) -> str:
        return str(self.multi_face_policy.currentData() or MULTI_FACE_REQUIRE_ONE)

    def _multi_face_policy_changed(self) -> None:
        self._clear_gallery_embedding_cache()
        if hasattr(self, "result_table"):
            self.results = []
            self.result_table.setRowCount(0)
        if hasattr(self, "query_input"):
            self.query_input.set_faces([])
        self.set_status("Multi-face handling changed. Gallery embeddings will be recomputed.")
        self._update_policy_notice()

    def refresh(self) -> None:
        engine_identity = self._engine_cache_identity(self.context.engine)
        if engine_identity != self._active_engine_cache_identity:
            self._active_engine_cache_identity = engine_identity
            self._clear_gallery_embedding_cache()
            self.results = []
            self.result_table.setRowCount(0)
            self.query_input.set_faces([])
        self._update_policy_notice()

    @staticmethod
    def _engine_cache_identity(engine) -> tuple[object, ...]:
        try:
            model_dir = str(engine.resolve_model_dir())
        except Exception:
            model_dir = str(getattr(engine, "custom_model_dir", "") or "")
        return (
            id(engine),
            str(getattr(engine, "model_name", "")),
            model_dir,
            tuple(getattr(engine, "requested_providers", ())),
            tuple(getattr(engine, "det_size", ())),
        )

    def _update_policy_notice(self) -> None:
        language = self.context.config.ui_language
        self.workflow_notice.setText(
            tr(
                "Gallery face embeddings are cached in memory after the first run and reused while the gallery image list "
                "and multi-face policy are unchanged. Changing only the query image reuses the cached gallery; adding, "
                "removing, clearing gallery images, changing multi-face handling, or changing the global model clears "
                "and recomputes it.",
                language,
            )
            + " "
            + tr(multi_face_policy_help(self._multi_face_policy()), language)
        )
