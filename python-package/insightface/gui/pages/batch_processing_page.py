"""Batch folder processing page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QLabel, QProgressBar, QSpinBox, QTableWidget, QTableWidgetItem

from ..core.exporters import export_csv, export_json
from ..core.utils import list_images, read_image, save_image, timestamp_for_filename
from ..widgets.drop_input import DropInput
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage


class BatchProcessingPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Batch Folder Processing", "Scan a folder, detect faces, optionally identify people, and export batch results.", parent)
        self.folder = ""
        self.rows = []
        self.recursive = QCheckBox("Recursive")
        self.recursive.setChecked(True)
        self.identify = QCheckBox("Identify people")
        self.identify.setChecked(True)
        self.save_crops = QCheckBox("Save face crops")
        self.save_crops.setChecked(context.config.save_crops)
        self.min_score = QSpinBox()
        self.min_score.setRange(0, 100)
        self.min_score.setValue(int(context.config.min_detection_score * 100))
        self.progress = QProgressBar()
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(["image_path", "face_index", "person_id", "person_name", "similarity", "threshold", "status", "bbox", "det_score", "crop_path"])
        configure_table_columns(self.table, [300, 80, 80, 160, 90, 90, 110, 220, 90, 260])
        self.folder_input = DropInput("Image Folder", mode="folder")
        self.folder_input.pathsChanged.connect(lambda paths: self.set_folder(paths[0]) if paths else self.clear_folder())
        self.content.addWidget(self.folder_input)
        self.content.addWidget(self.row(self.recursive, self.identify, self.save_crops, QLabel("Min det score %"), self.min_score, self.button("Start", self.start), self.button("Export", self.export)))
        self.content.addWidget(self.progress)
        self.content.addWidget(self.table, 1)

    def set_folder(self, folder: str) -> None:
        self.folder = folder
        self.set_status(f"Selected folder: {folder}")

    def clear_folder(self) -> None:
        self.folder = ""
        self.rows = []
        self.progress.setValue(0)
        self.table.setRowCount(0)

    def start(self) -> None:
        if not self.folder:
            self.show_error("Select a folder first.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        paths = [str(path) for path in list_images(self.folder, recursive=self.recursive.isChecked())]
        min_score = self.min_score.value() / 100.0
        model_name = self.context.engine.model_name

        def task(progress=None, is_cancelled=None):
            rows = []
            gallery = (
                self.context.storage.load_all_gallery_embeddings(
                    model_name=model_name
                )
                if self.identify.isChecked()
                else []
            )
            for index, path in enumerate(paths):
                if is_cancelled and is_cancelled():
                    break
                image = read_image(path)
                if image is None:
                    rows.append({"image_path": path, "status": "error", "error": "image read failure"})
                    continue
                media_id = self.context.storage.add_media_item(path, "image", width=image.shape[1], height=image.shape[0], processed_at=timestamp_for_filename())
                faces = [face for face in self.context.engine.detect_faces(image, source_path=path) if face.det_score >= min_score]
                for face_index, face in enumerate(faces):
                    result = None
                    if face.normed_embedding is not None and gallery:
                        matches = self.context.storage.search_embeddings(
                            face.normed_embedding,
                            self.context.config.default_top_k,
                            self.context.config.recognition_threshold,
                            model_name=model_name,
                        )
                        result = matches[0] if matches else None
                    person_name = result.person_name if result and result.similarity >= self.context.config.recognition_threshold else "Unknown"
                    similarity = result.similarity if result else 0.0
                    status = "matched" if result and result.similarity >= self.context.config.recognition_threshold else "unknown"
                    crop_path = ""
                    if self.save_crops.isChecked() and face.crop is not None:
                        crop_path = str(Path(self.context.config.crop_dir) / f"batch_{media_id}_{face_index}_{timestamp_for_filename()}.png")
                        save_image(crop_path, face.crop)
                    self.context.storage.add_media_face(media_id, face.normed_embedding, crop_path=crop_path, bbox=face.bbox, kps=face.kps, det_score=face.det_score, predicted_person_id=result.person_id if result else None, similarity=similarity, status=status)
                    rows.append({"image_path": path, "face_index": face_index, "person_id": result.person_id if result else "", "person_name": person_name, "similarity": similarity, "threshold": self.context.config.recognition_threshold, "status": status, "bbox": face.bbox, "det_score": face.det_score, "crop_path": crop_path})
                if progress:
                    progress(index + 1, len(paths), f"Processed {index + 1}/{len(paths)} files")
            return rows

        def done(rows):
            self.rows = rows
            self.progress.setValue(100)
            self._populate()
            self.set_status(f"Batch complete. {len(rows)} face row(s).")

        self.run_task("Batch folder processing", task, done)

    def _populate(self) -> None:
        columns = ["image_path", "face_index", "person_id", "person_name", "similarity", "threshold", "status", "bbox", "det_score", "crop_path"]
        self.table.setRowCount(len(self.rows))
        for row_index, row in enumerate(self.rows):
            for col, key in enumerate(columns):
                value = row.get(key, "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                self.table.setItem(row_index, col, QTableWidgetItem(str(value)))
        refresh_table_columns(self.table)

    def export(self) -> None:
        if not self.rows:
            self.show_error("No batch results to export.")
            return
        path = Path(self.context.config.export_dir) / f"batch_results_{timestamp_for_filename()}.csv"
        export_csv(path, self.rows)
        export_json(path.with_suffix(".json"), self.rows)
        self.set_status(f"Batch results exported to {path}")
