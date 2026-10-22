"""1:N face search page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QInputDialog, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ..core.exporters import export_csv, export_json
from ..core.utils import read_image, save_image, timestamp_for_filename
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from ..widgets.threshold_slider import ThresholdSlider
from ..widgets.upload_preview import UploadPreview
from .base import BasePage


class FaceSearchPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "1:N Face Search", "Search a query face against the local People Library.", parent)
        self.query_path = ""
        self.query_image = None
        self.query_face = None
        self.query_model_name = ""
        self.query_provider = ""
        self.results = []
        self.query_input = UploadPreview(
            "Query Image",
            extensions=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            dialog_filter="Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)",
        )
        self.query_input.pathChanged.connect(lambda path: self.load_query(path) if path else self.clear_query())
        self.content.addWidget(
            self.row(
                self.button("Search", self.search),
                self.button("Add as New Person", self.add_as_new_person),
                self.button("Assign to Existing Person", self.assign_existing),
                self.button("Export Results", self.export_results),
            )
        )
        self.threshold = ThresholdSlider(context.config.recognition_threshold)
        self.content.addWidget(self.threshold)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.viewer = self.query_input
        left_layout.addWidget(self.viewer)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.table = QTableWidget(0, 8)
        self.table.setIconSize(QSize(48, 48))
        self.table.setHorizontalHeaderLabels(["rank", "person_id", "thumbnail", "person_name", "similarity", "status", "sample_id", "crop_path"])
        configure_table_columns(self.table, [60, 90, 90, 190, 90, 110, 90, 280])
        right_layout.addWidget(self.table)
        splitter.addWidget(left)
        splitter.addWidget(right)
        self.content.addWidget(splitter, 1)

    def load_query(self, path: str) -> None:
        self.query_path = path
        self.query_image = read_image(path)
        if self.query_image is None:
            self.viewer.clear(emit=False)
            self.show_error("Image read failure.")
            return
        self.viewer.set_image(self.query_image, path)

    def clear_query(self) -> None:
        self.query_path = ""
        self.query_image = None
        self.query_face = None
        self.query_model_name = ""
        self.query_provider = ""
        self.results = []
        self.viewer.clear(emit=False)
        self.table.setRowCount(0)

    def search(self) -> None:
        if self.query_image is None:
            self.show_error("Please select a query image.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        model_name = self.context.engine.model_name
        provider = self.context.config.provider
        if not self.context.storage.load_all_gallery_embeddings(
            model_name=model_name
        ):
            self.show_error("No registered people. Please add people first.")
            return

        def task():
            face = self.context.engine.detect_best_face(self.query_image, source_path=self.query_path)
            if face is None or face.normed_embedding is None:
                raise ValueError("No face detected or embedding unavailable.")
            results = self.context.storage.search_embeddings(
                face.normed_embedding,
                top_k=self.context.config.default_top_k,
                threshold=self.threshold.value(),
                model_name=model_name,
            )
            return face, results

        def done(payload):
            self.query_face, self.results = payload
            self.query_model_name = model_name
            self.query_provider = provider
            self.viewer.set_faces([{"bbox": self.query_face.bbox, "label": "Query"}])
            self.table.setRowCount(len(self.results))
            for row, result in enumerate(self.results):
                data = [
                    row + 1,
                    result.person_id if result.person_id is not None else "",
                    "",
                    result.person_name,
                    f"{result.similarity:.4f}",
                    result.status,
                    result.sample_id,
                    result.crop_path or "",
                ]
                for col, value in enumerate(data):
                    item = QTableWidgetItem(str(value))
                    if col == 2:
                        icon = self._thumbnail_icon(result.crop_path)
                        if icon is not None:
                            item.setIcon(icon)
                    self.table.setItem(row, col, item)
                self.table.setRowHeight(row, 54)
            refresh_table_columns(self.table)
            self.set_status(f"Search complete. {len(self.results)} result(s).")

        self.run_task("Searching people library", task, done)

    def add_as_new_person(self) -> None:
        if self.query_face is None or self.query_face.normed_embedding is None:
            self.show_error("Run a search first.")
            return
        name, ok = QInputDialog.getText(self, "Add as New Person", "Person name")
        if not ok or not name.strip():
            return
        person_id = self.context.storage.add_person(name.strip())
        crop_path = Path(self.context.config.crop_dir) / f"person_{person_id}_{timestamp_for_filename()}.png"
        if self.query_face.crop is not None:
            save_image(crop_path, self.query_face.crop)
        self.context.storage.add_face_sample(
            person_id,
            self.query_face.normed_embedding,
            source_image_path=self.query_path,
            crop_path=str(crop_path),
            bbox=self.query_face.bbox,
            kps=self.query_face.kps,
            det_score=self.query_face.det_score,
            quality_score=self.query_face.quality_score,
            model_name=self.query_model_name,
            provider=self.query_provider,
        )
        self.set_status(f"Added {name.strip()} to People Library.")

    def assign_existing(self) -> None:
        people = self.context.storage.list_people()
        if self.query_face is None or self.query_face.normed_embedding is None:
            self.show_error("Run a search first.")
            return
        if not people:
            self.show_error("No registered people. Please add people first.")
            return
        names = [person["display_name"] or person["name"] for person in people]
        name, ok = QInputDialog.getItem(self, "Assign to Existing Person", "Person", names, 0, False)
        if not ok:
            return
        person = people[names.index(name)]
        self.context.storage.add_face_sample(
            int(person["id"]),
            self.query_face.normed_embedding,
            source_image_path=self.query_path,
            bbox=self.query_face.bbox,
            kps=self.query_face.kps,
            det_score=self.query_face.det_score,
            quality_score=self.query_face.quality_score,
            model_name=self.query_model_name,
            provider=self.query_provider,
        )
        self.set_status(f"Assigned query face to {name}.")

    def export_results(self) -> None:
        if not self.results:
            self.show_error("No search results to export.")
            return
        path = Path(self.context.config.export_dir) / f"face_search_{timestamp_for_filename()}.csv"
        rows = [result.to_json_dict() for result in self.results]
        export_csv(path, rows)
        export_json(path.with_suffix(".json"), rows)
        self.set_status(f"Search results exported to {path}")

    @staticmethod
    def _thumbnail_icon(path: str | None) -> QIcon | None:
        if not path:
            return None
        pixmap = QPixmap(str(Path(path)))
        if pixmap.isNull():
            return None
        return QIcon(pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
