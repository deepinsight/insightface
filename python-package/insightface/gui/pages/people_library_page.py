"""People library management page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QCursor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.exporters import export_csv
from ..core.utils import list_images, read_image, save_image, timestamp_for_filename
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)"


class ImageOrFolderImportBox(QFrame):
    pathsSelected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageOrFolderImport")
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setProperty("hoverActive", False)
        self.setProperty("dragActive", False)
        self.setProperty("hasFiles", False)
        layout = QVBoxLayout(self)
        title = QLabel("Import Image or Folder")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:15px; font-weight:700;")
        hint = QLabel("Click to import an image or folder, or drag one here. Folders are imported recursively.")
        hint.setObjectName("dropPrompt")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        self.button = QPushButton("Import Image or Folder")
        self.button.clicked.connect(self._show_menu)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(self.button, 0, Qt.AlignHCenter)
        for watched in (self, title, hint):
            watched.installEventFilter(self)

    def _show_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Select Image", self._select_image)
        menu.addAction("Select Folder", self._select_folder)
        menu.exec(self.button.mapToGlobal(self.button.rect().bottomLeft()))

    def _select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select image", str(Path.home()), IMAGE_FILTER)
        if path:
            self.pathsSelected.emit([path])

    def _select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder", str(Path.home()))
        if folder:
            self.pathsSelected.emit([str(path) for path in list_images(folder, recursive=True)])

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and self._accepted_urls(event.mimeData().urls()):
            self._set_drag_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self._set_drag_active(False)
        paths: list[str] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                paths.extend(str(item) for item in list_images(path, recursive=True))
            elif path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(path))
        if paths:
            self.pathsSelected.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _accepted_urls(self, urls) -> bool:
        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir() or path.suffix.lower() in IMAGE_EXTENSIONS:
                return True
        return False

    def _set_drag_active(self, value: bool) -> None:
        self._set_property("dragActive", value)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Enter:
            self._set_property("hoverActive", True)
            return False
        if event.type() == QEvent.Leave:
            self._update_hover_from_cursor()
            return False
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            self._show_menu()
            return True
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


class PeopleLibraryPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "People Library",
            "Build the local gallery used by 1:N Face Search. Imported images create people named from the file name by default.",
            parent,
        )
        self.selected_person_id = None
        self.people = []
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.import_box = ImageOrFolderImportBox()
        self.import_box.pathsSelected.connect(self.import_paths)
        left_layout.addWidget(self.import_box)
        left_layout.addWidget(QLabel("People"))
        self.people_table = QTableWidget(0, 3)
        self.people_table.setHorizontalHeaderLabels(["ID", "Thumbnail", "Name"])
        self.people_table.setIconSize(QSize(48, 48))
        configure_table_columns(self.people_table, [60, 90, 240])
        self.people_table.currentCellChanged.connect(self.on_person_selected)
        left_layout.addWidget(self.people_table, 1)
        left_layout.addWidget(self.row(self.button("Add Person", self.add_person), self.button("Delete", self.delete_person), self.button("Export Summary", self.export_summary)))

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.details = QTextEdit()
        self.details.setPlaceholderText("Name, notes, and tags are shown here.")
        right_layout.addWidget(self.details)
        right_layout.addWidget(self.row(self.button("Save Details", self.save_details)))
        self.samples_table = QTableWidget(0, 7)
        self.samples_table.setHorizontalHeaderLabels(["id", "crop_path", "det_score", "quality_score", "model_name", "provider", "source_image_path"])
        configure_table_columns(self.samples_table, [60, 260, 90, 100, 130, 110, 320])
        right_layout.addWidget(self.samples_table)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.content.addWidget(splitter, 1)
        self.refresh()

    def refresh(self) -> None:
        current_id = self.selected_person_id
        self.people = self.context.storage.list_people()
        self.people_table.setRowCount(len(self.people))
        for row, person in enumerate(self.people):
            id_item = QTableWidgetItem(str(person["id"]))
            id_item.setData(Qt.UserRole, person["id"])
            thumb_item = QTableWidgetItem()
            icon = self._thumbnail_icon(person.get("cover_crop_path"))
            if icon is not None:
                thumb_item.setIcon(icon)
            name_item = QTableWidgetItem(person["display_name"] or person["name"])
            name_item.setData(Qt.UserRole, person["id"])
            self.people_table.setItem(row, 0, id_item)
            self.people_table.setItem(row, 1, thumb_item)
            self.people_table.setItem(row, 2, name_item)
            self.people_table.setRowHeight(row, 54)
            if current_id == person["id"]:
                self.people_table.selectRow(row)
        refresh_table_columns(self.people_table)
        if current_id and any(person["id"] == current_id for person in self.people):
            self.selected_person_id = current_id
        elif self.people:
            self.people_table.selectRow(0)
            self.selected_person_id = self.people[0]["id"]
        else:
            self.selected_person_id = None
        self._populate_details()
        self.refresh_samples()

    def on_person_selected(self, current_row: int, current_column: int, previous_row: int, previous_column: int) -> None:
        del current_column, previous_row, previous_column
        if current_row < 0 or current_row >= len(self.people):
            self.selected_person_id = None
        else:
            self.selected_person_id = int(self.people[current_row]["id"])
        self._populate_details()
        self.refresh_samples()

    def _populate_details(self) -> None:
        person = next((row for row in self.people if row["id"] == self.selected_person_id), None)
        if not person:
            self.details.clear()
            return
        self.details.setPlainText(
            "\n".join(
                [
                    f"name: {person['name']}",
                    f"display_name: {person['display_name'] or ''}",
                    f"tags: {person['tags'] or ''}",
                    "notes:",
                    person["notes"] or "",
                ]
            )
        )

    def add_person(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Person", "Person name")
        if ok and name.strip():
            self.context.storage.add_person(name.strip())
            self.refresh()

    def delete_person(self) -> None:
        if self.selected_person_id is None:
            self.show_error("Select a person first.")
            return
        self.context.storage.delete_person(int(self.selected_person_id), delete_samples=False)
        self.selected_person_id = None
        self.refresh()

    def save_details(self) -> None:
        if self.selected_person_id is None:
            self.show_error("Select a person first.")
            return
        fields = {}
        lines = self.details.toPlainText().splitlines()
        notes_start = None
        for idx, line in enumerate(lines):
            if line.startswith("name:"):
                fields["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("display_name:"):
                fields["display_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("tags:"):
                fields["tags"] = line.split(":", 1)[1].strip()
            elif line.strip() == "notes:":
                notes_start = idx + 1
        if notes_start is not None:
            fields["notes"] = "\n".join(lines[notes_start:]).strip()
        self.context.storage.update_person(int(self.selected_person_id), **fields)
        self.refresh()
        self.set_status("Person details saved.")

    def import_paths(self, paths: list[str]) -> None:
        paths = [str(Path(path)) for path in paths if Path(path).suffix.lower() in IMAGE_EXTENSIONS]
        if not paths:
            self.show_error("No supported images found.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        model_name = self.context.engine.model_name
        provider = self.context.config.provider

        def task(progress=None, is_cancelled=None):
            imported = 0
            skipped = 0
            for index, path in enumerate(paths):
                if is_cancelled and is_cancelled():
                    break
                image = read_image(path)
                if image is None:
                    skipped += 1
                    if progress:
                        progress(index + 1, len(paths), f"Skipped unreadable image: {Path(path).name}")
                    continue
                face = self.context.engine.detect_best_face(image, source_path=path)
                if face is None or face.normed_embedding is None:
                    skipped += 1
                    if progress:
                        progress(index + 1, len(paths), f"No usable face: {Path(path).name}")
                    continue
                person_id = self.context.storage.add_person(Path(path).stem)
                crop_path = Path(self.context.config.crop_dir) / f"person_{person_id}_{timestamp_for_filename()}_{index}.png"
                if face.crop is not None:
                    save_image(crop_path, face.crop)
                self.context.storage.add_face_sample(
                    person_id,
                    face.normed_embedding,
                    source_image_path=path,
                    crop_path=str(crop_path),
                    bbox=face.bbox,
                    kps=face.kps,
                    det_score=face.det_score,
                    quality_score=face.quality_score,
                    model_name=model_name,
                    provider=provider,
                )
                imported += 1
                if progress:
                    progress(index + 1, len(paths), f"Imported {imported} people, skipped {skipped}")
            return {"imported": imported, "skipped": skipped}

        def done(result):
            self.refresh()
            self.set_status(f"Imported {result['imported']} people. Skipped {result['skipped']} image(s).")

        self.run_task("Importing People Library", task, done)

    def refresh_samples(self) -> None:
        if self.selected_person_id is None:
            self.samples_table.setRowCount(0)
            return
        samples = self.context.storage.list_face_samples(self.selected_person_id)
        self.samples_table.setRowCount(len(samples))
        columns = ["id", "crop_path", "det_score", "quality_score", "model_name", "provider", "source_image_path"]
        for row, sample in enumerate(samples):
            for col, key in enumerate(columns):
                self.samples_table.setItem(row, col, QTableWidgetItem(str(sample.get(key, ""))))
        refresh_table_columns(self.samples_table)

    def export_summary(self) -> None:
        path = Path(self.context.config.export_dir) / f"people_summary_{timestamp_for_filename()}.csv"
        export_csv(path, self.context.storage.list_people())
        self.set_status(f"People summary exported to {path}")

    @staticmethod
    def _thumbnail_icon(path: str | None) -> QIcon | None:
        if not path:
            return None
        pixmap = QPixmap(str(Path(path)))
        if pixmap.isNull():
            return None
        return QIcon(pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
