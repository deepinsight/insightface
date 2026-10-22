"""Multi-face photo recognition page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ..core.exporters import export_annotated_image, export_csv, export_json
from ..core.utils import read_image, save_image, timestamp_for_filename
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from ..widgets.upload_preview import UploadPreview
from .base import BasePage


class MultiFacePhotoPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Multi-face Photo Recognition", "Detect all faces in a group photo and match them against People Library.", parent)
        self.image_path = ""
        self.image = None
        self.rows = []
        self.photo_input = UploadPreview(
            "Photo",
            extensions=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            dialog_filter="Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)",
        )
        self.photo_input.pathChanged.connect(lambda path: self.load_photo(path) if path else self.clear_photo())
        self.content.addWidget(
            self.row(
                self.button("Recognize Faces", self.recognize),
                self.button("Export Annotated Image", self.export_image),
                self.button("Export CSV / JSON", self.export_rows),
            )
        )
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.viewer = self.photo_input
        left_layout.addWidget(self.viewer)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["face_index", "person_name", "similarity", "status", "bbox", "det_score"])
        configure_table_columns(self.table, [90, 180, 90, 110, 230, 90])
        right_layout.addWidget(self.table)
        splitter.addWidget(left)
        splitter.addWidget(right)
        self.content.addWidget(splitter, 1)

    def load_photo(self, path: str) -> None:
        self.image_path = path
        self.image = read_image(path)
        if self.image is None:
            self.viewer.clear(emit=False)
            self.show_error("Image read failure.")
            return
        self.viewer.set_image(self.image, path)

    def clear_photo(self) -> None:
        self.image_path = ""
        self.image = None
        self.rows = []
        self.viewer.clear(emit=False)
        self.table.setRowCount(0)

    def recognize(self) -> None:
        if self.image is None:
            self.show_error("Please select a photo.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        model_name = self.context.engine.model_name

        def task():
            faces = self.context.engine.detect_faces(self.image, source_path=self.image_path)
            gallery = self.context.storage.load_all_gallery_embeddings(
                model_name=model_name
            )
            rows = []
            overlays = []
            media_id = self.context.storage.add_media_item(self.image_path, "image", width=self.image.shape[1], height=self.image.shape[0], processed_at=timestamp_for_filename())
            for index, face in enumerate(faces):
                result = None
                if face.normed_embedding is not None and gallery:
                    result_list = self.context.storage.search_embeddings(
                        face.normed_embedding,
                        self.context.config.default_top_k,
                        self.context.config.recognition_threshold,
                        model_name=model_name,
                    )
                    result = result_list[0] if result_list else None
                name = result.person_name if result and result.similarity >= self.context.config.recognition_threshold else "Unknown"
                similarity = result.similarity if result else 0.0
                status = "matched" if result and result.similarity >= self.context.config.recognition_threshold else "unknown"
                crop_path = ""
                if face.crop is not None:
                    crop_path = str(Path(self.context.config.crop_dir) / f"media_{media_id}_{index}_{timestamp_for_filename()}.png")
                    save_image(crop_path, face.crop)
                self.context.storage.add_media_face(media_id, face.normed_embedding, crop_path=crop_path, bbox=face.bbox, kps=face.kps, det_score=face.det_score, predicted_person_id=result.person_id if result else None, similarity=similarity, status=status)
                row = {"face_index": index, "person_name": name, "similarity": similarity, "status": status, "bbox": face.bbox, "det_score": face.det_score, "crop_path": crop_path}
                rows.append(row)
                overlays.append({"bbox": face.bbox, "label": name, "similarity": similarity})
            return rows, overlays

        def done(payload):
            self.rows, overlays = payload
            self.viewer.set_faces(overlays)
            self.table.setRowCount(len(self.rows))
            columns = ["face_index", "person_name", "similarity", "status", "bbox", "det_score"]
            for row_index, row in enumerate(self.rows):
                for col, key in enumerate(columns):
                    value = row[key]
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    self.table.setItem(row_index, col, QTableWidgetItem(str(value)))
            refresh_table_columns(self.table)
            self.set_status(f"Recognized {len(self.rows)} face(s).")

        self.run_task("Recognizing photo", task, done)

    def export_image(self) -> None:
        image = self.viewer.render_with_overlay()
        if image is None:
            self.show_error("No annotated image to export.")
            return
        path = Path(self.context.config.export_dir) / f"annotated_{timestamp_for_filename()}.png"
        export_annotated_image(path, image)
        self.set_status(f"Annotated image exported to {path}")

    def export_rows(self) -> None:
        if not self.rows:
            self.show_error("No recognition results to export.")
            return
        path = Path(self.context.config.export_dir) / f"multiface_{timestamp_for_filename()}.csv"
        export_csv(path, self.rows)
        export_json(path.with_suffix(".json"), self.rows)
        self.set_status(f"Results exported to {path}")
