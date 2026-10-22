"""Source + Target face swap page for images and videos."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

from ..core.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from ..core.face_engine import providers_from_choice
from ..core.i18n import tr
from ..core.model_downloads import list_installed_swap_models
from ..core.swap import FaceSwapEngine
from ..core.utils import read_image, save_image, timestamp_for_filename
from ..widgets.image_viewer import ImageViewer
from ..widgets.upload_preview import UploadPreview
from .base import BasePage


class FaceSwapPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Face Swap",
            "Source + Target = Result. Target can be an image or a video; the workflow chooses image or video swap automatically.",
            parent,
        )
        self.source_path = ""
        self.target_path = ""
        self.target_kind = ""
        self.output_image = None
        self.output_video_path = ""
        self.output_path = ""
        self.source_image = None
        self.target_image = None
        self.content.addWidget(
            self.notice(
                "Face swap may require separate commercial authorization depending on usage and model license. "
                "Use only with appropriate rights and consent."
            )
        )

        image_filter = "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)"
        target_filter = "Images and videos (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All Files (*)"
        self.content.addWidget(
            self.row(
                self.button("Run Face Swap", self.run_swap),
                self.button("Open Result Directory", self.open_result_directory),
            )
        )
        splitter = QSplitter(Qt.Horizontal)
        self.source_view = UploadPreview("Source", extensions=IMAGE_EXTENSIONS, dialog_filter=image_filter)
        self.target_view = UploadPreview("Target", extensions=IMAGE_EXTENSIONS | VIDEO_EXTENSIONS, dialog_filter=target_filter)
        self.output_view = ImageViewer()
        self.output_view.setToolTip("Click to open the saved result file.")
        self.output_view.viewport().installEventFilter(self)
        self.output_view.installEventFilter(self)
        self.result_label = QLabel("Result preview appears here. Video results are saved to the exports folder.")
        self.result_label.setWordWrap(True)
        self.result_label.setProperty("role", "muted")
        self.source_view.pathChanged.connect(lambda path: self.load_source(path) if path else self.clear_source())
        self.target_view.pathChanged.connect(lambda path: self.load_target(path) if path else self.clear_target())

        for title, viewer in [("Source", self.source_view), ("Target", self.target_view)]:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            label = QLabel(title)
            label.setStyleSheet("font-weight:700;")
            layout.addWidget(label)
            layout.addWidget(viewer)
            splitter.addWidget(panel)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_title = QLabel("Result")
        result_title.setStyleSheet("font-weight:700;")
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.output_view)
        result_layout.addWidget(self.result_label)
        splitter.addWidget(result_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        self.content.addWidget(splitter, 1)

    def load_source(self, path: str) -> None:
        self.source_path = path
        self.source_image = read_image(path)
        if self.source_image is None:
            self.source_view.clear(emit=False)
            self.show_error("Image read failure.")
            return
        self._clear_output()
        self.source_view.set_image(self.source_image, path)

    def clear_source(self) -> None:
        self.source_path = ""
        self.source_image = None
        self.source_view.clear(emit=False)
        self._clear_output()

    def load_target(self, path: str) -> None:
        self.target_path = path
        suffix = Path(path).suffix.lower()
        self._clear_output()
        if suffix in IMAGE_EXTENSIONS:
            self.target_kind = "image"
            self.target_image = read_image(path)
            if self.target_image is None:
                self.target_view.clear(emit=False)
                self.show_error("Image read failure.")
                return
            self.target_view.set_image(self.target_image, path)
            self.set_status(f"Selected target image: {Path(path).name}")
            return
        if suffix in VIDEO_EXTENSIONS:
            self.target_kind = "video"
            self.target_image = self._read_video_preview(path)
            if self.target_image is not None:
                self.target_view.set_image(self.target_image, path)
            else:
                self.target_view.set_path(path, emit=False)
            self.set_status(f"Selected target video: {Path(path).name}")
            return
        self.target_kind = ""
        self.target_image = None
        self.target_view.clear(emit=False)
        self.show_error("Unsupported target file type.")

    def clear_target(self) -> None:
        self.target_path = ""
        self.target_kind = ""
        self.target_image = None
        self.target_view.clear(emit=False)
        self._clear_output()

    def run_swap(self) -> None:
        if self.source_image is None or not self.source_path:
            self.show_error("Select a source image first.")
            return
        if not self.target_path or self.target_kind not in {"image", "video"}:
            self.show_error("Select a target image or video first.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        model_path = self._resolve_swap_model_path()
        if not model_path:
            self.show_error("Face swap model not found. Please download and choose a swap model in Models.")
            return

        source_image = self.source_image.copy()
        target_path = self.target_path
        target_kind = self.target_kind
        target_image = self.target_image.copy() if self.target_image is not None and target_kind == "image" else None

        def task(progress=None, is_cancelled=None):
            swapper = FaceSwapEngine(
                model_path,
                providers_from_choice(self.context.config.provider),
                gfpgan_model_path=getattr(self.context.config, "gfpgan_model_path", ""),
                enable_gfpgan=bool(getattr(self.context.config, "enable_gfpgan", False)),
            )
            if not swapper.load():
                raise ValueError(swapper.last_error)
            source_face = self.context.engine.detect_best_face(source_image, source_path=self.source_path)
            if source_face is None or source_face.normed_embedding is None:
                raise ValueError("No usable face detected in source image.")
            source_native = SimpleNamespace(normed_embedding=source_face.normed_embedding)
            if target_kind == "image":
                return self._swap_image(swapper, source_native, target_image, target_path)
            return self._swap_video(swapper, source_native, target_path, progress, is_cancelled)

        def done(result):
            if result["kind"] == "image":
                self.output_image = result["image"]
                self.output_video_path = ""
                self.output_path = result["path"]
                self.output_view.set_image(self.output_image)
                self.result_label.setText(
                    tr("Image swap saved. Click Result to open it.", self.context.config.ui_language)
                    + f"\n{self.output_path}"
                )
            else:
                self.output_image = result.get("preview")
                self.output_video_path = result["path"]
                self.output_path = result["path"]
                if self.output_image is not None:
                    self.output_view.set_image(self.output_image)
                self.result_label.setText(
                    tr("Video swap saved. Click Result to open it.", self.context.config.ui_language)
                    + f"\n{self.output_video_path}"
                )
            self.set_status(result["message"])

        self.run_task("Running face swap", task, done)

    def open_result(self) -> None:
        if self.output_path and Path(self.output_path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_path))
            return
        self.show_error("No saved result file to open.")

    def open_result_directory(self) -> None:
        folder = Path(self.output_path).parent if self.output_path else Path(self.context.config.export_dir)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _swap_image(self, swapper: FaceSwapEngine, source_native, target_image, target_path: str) -> dict:
        if target_image is None:
            raise ValueError("Target image could not be read.")
        target_face = self.context.engine.detect_best_face(target_image, source_path=target_path)
        if target_face is None or target_face.kps is None:
            raise ValueError("No usable face detected in target image.")
        target_native = SimpleNamespace(kps=np.asarray(target_face.kps, dtype=np.float32))
        image = swapper.swap(target_image, target_native, source_native)
        output_path = Path(self.context.config.export_dir) / f"face_swap_{timestamp_for_filename()}.png"
        save_image(output_path, image)
        return {
            "kind": "image",
            "image": image,
            "path": str(output_path),
            "message": f"Image face swap saved to {output_path}",
        }

    def _swap_video(self, swapper: FaceSwapEngine, source_native, target_path: str, progress=None, is_cancelled=None) -> dict:
        try:
            import cv2
        except Exception as exc:
            raise ValueError(f"OpenCV is required for video face swap: {exc}") from exc

        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            raise ValueError("Video could not be opened.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Video dimensions could not be read.")
        output_path = Path(self.context.config.export_dir) / f"face_swap_video_{timestamp_for_filename()}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError("Video writer could not be opened.")

        preview = None
        swapped = 0
        processed = 0
        try:
            while True:
                if is_cancelled and is_cancelled():
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                output_frame = frame
                target_face = self.context.engine.detect_best_face(frame, source_path=target_path)
                if target_face is not None and target_face.kps is not None:
                    target_native = SimpleNamespace(kps=np.asarray(target_face.kps, dtype=np.float32))
                    output_frame = swapper.swap(frame, target_native, source_native)
                    swapped += 1
                    if preview is None:
                        preview = output_frame.copy()
                writer.write(output_frame)
                processed += 1
                if progress:
                    progress(processed, frame_count or processed, f"Processed {processed} frame(s), swapped {swapped}")
        finally:
            writer.release()
            cap.release()
        return {
            "kind": "video",
            "path": str(output_path),
            "preview": preview,
            "message": f"Video face swap complete. Processed {processed} frame(s), swapped {swapped} frame(s).",
        }

    def _resolve_swap_model_path(self) -> str:
        configured = getattr(self.context.config, "swap_model_path", "")
        if configured and Path(configured).expanduser().exists():
            return str(Path(configured).expanduser())
        installed = list_installed_swap_models(self.context.config.model_root)
        return str(installed[0]) if installed else ""

    def _clear_output(self) -> None:
        self.output_image = None
        self.output_video_path = ""
        self.output_path = ""
        self.output_view.set_image(None)
        self.result_label.setText(
            tr("Result preview appears here. Video results are saved to the exports folder.", self.context.config.ui_language)
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched in {self.output_view, self.output_view.viewport()} and event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self.open_result()
                return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _read_video_preview(path: str) -> np.ndarray | None:
        try:
            import cv2

            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            return frame if ok else None
        except Exception:
            return None
