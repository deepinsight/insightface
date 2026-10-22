"""Video person search page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QLabel, QSpinBox, QTableWidget, QTableWidgetItem

from ..core.exporters import export_csv
from ..core.recognition import search_gallery
from ..core.utils import save_image, timestamp_for_filename
from ..core.video import iter_video_frames, read_video_thumbnail, timestamp_hhmmss
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from ..widgets.upload_preview import UploadPreview
from .base import BasePage


class VideoSearchPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Video Person Search", "Search for a registered person in a local video using simple frame sampling.", parent)
        self.video_path = ""
        self.rows = []
        self.person_combo = QComboBox()
        self.interval = QSpinBox()
        self.interval.setRange(1, 300)
        self.interval.setValue(context.config.video_frame_interval)
        self.video_input = UploadPreview(
            "Video File",
            extensions=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"],
            dialog_filter="Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All Files (*)",
        )
        self.video_input.pathChanged.connect(lambda path: self.load_video(path) if path else self.clear_video())
        self.content.addWidget(self.video_input)
        self.content.addWidget(self.row(QLabel("Target person"), self.person_combo, QLabel("Frame interval"), self.interval, self.button("Refresh People", self.refresh), self.button("Start", self.start), self.button("Export CSV", self.export)))
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["video_path", "timestamp_ms", "timestamp_hhmmss", "frame_index", "person_name", "similarity", "bbox", "crop_path"])
        configure_table_columns(self.table, [300, 120, 140, 100, 160, 90, 200, 260])
        self.content.addWidget(self.table, 1)
        self.refresh()

    def refresh(self) -> None:
        self.person_combo.clear()
        for person in self.context.storage.list_people():
            self.person_combo.addItem(person["display_name"] or person["name"], person["id"])

    def load_video(self, path: str) -> None:
        self.video_path = path
        self.rows = []
        self.table.setRowCount(0)
        self.video_input.set_image(read_video_thumbnail(path), path)
        self.set_status(f"Selected video: {path}")

    def clear_video(self) -> None:
        self.video_path = ""
        self.rows = []
        self.table.setRowCount(0)
        self.video_input.clear(emit=False)

    def start(self) -> None:
        if not self.video_path:
            self.show_error("Select a video first.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        person_id = self.person_combo.currentData()
        model_name = self.context.engine.model_name
        gallery = [
            item
            for item in self.context.storage.load_all_gallery_embeddings(
                model_name=model_name
            )
            if item.get("person_id") == person_id
        ]
        if not gallery:
            self.show_error("Target person has no registered face samples.")
            return

        def task(progress=None, is_cancelled=None):
            rows = []
            for frame_index, timestamp_ms, frame in iter_video_frames(self.video_path, self.interval.value()):
                if is_cancelled and is_cancelled():
                    break
                for face in self.context.engine.detect_faces(frame, source_path=self.video_path):
                    if face.normed_embedding is None:
                        continue
                    matches = search_gallery(face.normed_embedding, gallery, top_k=1, threshold=self.context.config.recognition_threshold)
                    if matches and matches[0].similarity >= self.context.config.recognition_threshold:
                        crop_path = str(Path(self.context.config.crop_dir) / f"video_{frame_index}_{timestamp_for_filename()}.png")
                        if face.crop is not None:
                            save_image(crop_path, face.crop)
                        rows.append({"video_path": self.video_path, "timestamp_ms": timestamp_ms, "timestamp_hhmmss": timestamp_hhmmss(timestamp_ms), "frame_index": frame_index, "person_id": person_id, "person_name": matches[0].person_name, "similarity": matches[0].similarity, "bbox": face.bbox, "crop_path": crop_path})
                if progress:
                    progress(frame_index, frame_index + 1, f"Processed frame {frame_index}")
            return rows

        self.run_task("Video person search", task, lambda rows: (setattr(self, "rows", rows), self._populate(), self.set_status(f"Video search complete. {len(rows)} hit(s).")))

    def _populate(self) -> None:
        columns = ["video_path", "timestamp_ms", "timestamp_hhmmss", "frame_index", "person_name", "similarity", "bbox", "crop_path"]
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
            self.show_error("No video results to export.")
            return
        path = Path(self.context.config.export_dir) / f"video_search_{timestamp_for_filename()}.csv"
        export_csv(path, self.rows)
        self.set_status(f"Video results exported to {path}")
