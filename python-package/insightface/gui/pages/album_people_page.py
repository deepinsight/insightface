"""Album people clustering page."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PySide6.QtWidgets import QLabel, QDoubleSpinBox, QTableWidget, QTableWidgetItem

from ..core.clustering import cluster_embeddings_dbscan
from ..core.exporters import export_csv
from ..core.utils import list_images, read_image, timestamp_for_filename
from ..widgets.drop_input import DropInput
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage


class AlbumPeoplePage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Album People Clustering", "Approximate local person grouping for album review and correction.", parent)
        self.folder = ""
        self.rows = []
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 0.99)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.28)
        self.content.addWidget(self.notice("Clustering is approximate. Please review and correct groups manually."))
        self.folder_input = DropInput("Album Folder", mode="folder")
        self.folder_input.pathsChanged.connect(lambda paths: self.set_folder(paths[0]) if paths else self.clear_folder())
        self.content.addWidget(self.folder_input)
        self.content.addWidget(self.row(QLabel("DBSCAN distance threshold"), self.threshold, self.button("Scan and Cluster", self.scan), self.button("Export CSV", self.export)))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["cluster_id", "face_count", "photo_count", "average_quality", "sample_paths"])
        configure_table_columns(self.table, [100, 100, 100, 120, 360])
        self.content.addWidget(self.table, 1)

    def set_folder(self, folder: str) -> None:
        self.folder = folder
        self.set_status(f"Selected album: {folder}")

    def clear_folder(self) -> None:
        self.folder = ""
        self.rows = []
        self.table.setRowCount(0)

    def scan(self) -> None:
        if not self.folder:
            self.show_error("Select an album folder first.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        paths = list_images(self.folder, recursive=True)

        def task(progress=None, is_cancelled=None):
            faces = []
            embeddings = []
            for index, path in enumerate(paths):
                if is_cancelled and is_cancelled():
                    break
                image = read_image(path)
                if image is None:
                    continue
                for face in self.context.engine.detect_faces(image, source_path=str(path)):
                    if face.normed_embedding is not None:
                        embeddings.append(face.normed_embedding)
                        faces.append({"path": str(path), "quality": face.quality_score or 0.0, "bbox": face.bbox})
                if progress:
                    progress(index + 1, len(paths), f"Scanned {index + 1}/{len(paths)} photos")
            labels, _algorithm = cluster_embeddings_dbscan(embeddings, distance_threshold=self.threshold.value()) if embeddings else ([], "none")
            clusters = defaultdict(list)
            for face, label in zip(faces, labels):
                clusters[int(label)].append(face)
            rows = []
            for label, items in clusters.items():
                rows.append({"cluster_id": label, "face_count": len(items), "photo_count": len({item["path"] for item in items}), "average_quality": sum(item["quality"] for item in items) / max(1, len(items)), "sample_paths": "; ".join(item["path"] for item in items[:5])})
            return rows

        self.run_task("Album clustering", task, lambda rows: (setattr(self, "rows", rows), self._populate(), self.set_status(f"Created {len(rows)} cluster(s).")))

    def _populate(self) -> None:
        columns = ["cluster_id", "face_count", "photo_count", "average_quality", "sample_paths"]
        self.table.setRowCount(len(self.rows))
        for row_index, row in enumerate(self.rows):
            for col, key in enumerate(columns):
                value = row[key]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                self.table.setItem(row_index, col, QTableWidgetItem(str(value)))
        refresh_table_columns(self.table)

    def export(self) -> None:
        if not self.rows:
            self.show_error("No clusters to export.")
            return
        path = Path(self.context.config.export_dir) / f"album_clusters_{timestamp_for_filename()}.csv"
        export_csv(path, self.rows)
        self.set_status(f"Clusters exported to {path}")
