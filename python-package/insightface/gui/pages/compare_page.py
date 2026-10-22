"""1:1 face compare page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QSplitter, QTextEdit, QWidget, QVBoxLayout
from PySide6.QtCore import Qt

from ..core.exporters import export_json, export_markdown
from ..core.utils import read_image, timestamp_for_filename
from ..widgets.threshold_slider import ThresholdSlider
from ..widgets.upload_preview import UploadPreview
from .base import BasePage


class ComparePage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "1:1 Face Compare", "Compare two local images and export a structured result.", parent)
        self.path_a = ""
        self.path_b = ""
        self.image_a = None
        self.image_b = None
        self._compare_generation = 0
        self._compare_running = False
        image_filter = "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)"
        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        self.input_a = UploadPreview("Image A", extensions=image_exts, dialog_filter=image_filter)
        self.input_b = UploadPreview("Image B", extensions=image_exts, dialog_filter=image_filter)
        self.input_a.pathChanged.connect(lambda path: self.load_a(path) if path else self.clear_a())
        self.input_b.pathChanged.connect(lambda path: self.load_b(path) if path else self.clear_b())
        self.compare_button = self.button("Compare", self.compare)
        controls = self.row(
            self.compare_button,
            self.button("Reset", self.reset),
            self.button("Export Result", self.export_result),
        )
        self.content.addWidget(controls)
        self.threshold = ThresholdSlider(context.config.recognition_threshold)
        self.content.addWidget(self.threshold)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.viewer_a = self.input_a
        left_layout.addWidget(self.viewer_a)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.viewer_b = self.input_b
        right_layout.addWidget(self.viewer_b)
        splitter.addWidget(left)
        splitter.addWidget(right)
        self.content.addWidget(splitter, 1)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.content.addWidget(self.result_text)
        self.result = None

    def load_a(self, path: str) -> None:
        self.path_a = path
        self.image_a = read_image(path)
        if self.image_a is None:
            self.viewer_a.clear(emit=False)
            self.show_error("Image read failure.")
            return
        self.viewer_a.set_image(self.image_a, path)
        self._invalidate_result("Image A replaced.")

    def load_b(self, path: str) -> None:
        self.path_b = path
        self.image_b = read_image(path)
        if self.image_b is None:
            self.viewer_b.clear(emit=False)
            self.show_error("Image read failure.")
            return
        self.viewer_b.set_image(self.image_b, path)
        self._invalidate_result("Image B replaced.")

    def clear_a(self) -> None:
        self.path_a = ""
        self.image_a = None
        self.viewer_a.set_image(None)
        self._invalidate_result()

    def clear_b(self) -> None:
        self.path_b = ""
        self.image_b = None
        self.viewer_b.set_image(None)
        self._invalidate_result()

    def compare(self) -> None:
        if self._compare_running:
            self.set_status("Compare is already running.")
            return
        if self.image_a is None or self.image_b is None:
            self.show_error("Please select both images first.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        image_a = self.image_a.copy()
        image_b = self.image_b.copy()
        path_a = self.path_a
        path_b = self.path_b
        threshold = self.threshold.value()
        self._compare_generation += 1
        generation = self._compare_generation
        self._compare_running = True
        self.compare_button.setEnabled(False)

        def task():
            try:
                result = self.context.engine.compare_images(
                    image_a,
                    image_b,
                    threshold=threshold,
                    path1=path_a,
                    path2=path_b,
                )
                return {"generation": generation, "result": result, "error": ""}
            except Exception as exc:
                return {"generation": generation, "result": None, "error": str(exc)}

        def done(payload):
            self._compare_running = False
            self.compare_button.setEnabled(True)
            if payload["generation"] != self._compare_generation:
                self.set_status("Ignored stale compare result.")
                return
            if payload["error"]:
                self.show_error(payload["error"])
                return
            result = payload["result"]
            self.result = result
            self.viewer_a.set_faces([{"bbox": result.face_a.bbox, "label": "Face A"}])
            self.viewer_b.set_faces([{"bbox": result.face_b.bbox, "label": "Face B"}])
            self.result_text.setPlainText(
                "\n".join(
                    [
                        f"Similarity: {result.similarity:.4f}",
                        f"Threshold: {result.threshold:.4f}",
                        f"Decision: {result.decision}",
                        f"Det score A: {result.face_a.det_score:.4f}",
                        f"Det score B: {result.face_b.det_score:.4f}",
                        "Notes: " + ("; ".join(result.notes) if result.notes else "Both faces detected successfully."),
                    ]
                )
            )
            self.set_status(f"Compare complete: {result.decision}")

        self.run_task("Comparing faces", task, done, show_dialog=False)

    def reset(self) -> None:
        self.path_a = self.path_b = ""
        self.image_a = self.image_b = None
        self.result = None
        self.input_a.clear(emit=False)
        self.input_b.clear(emit=False)
        self._invalidate_result("Compare page reset.")
        self._compare_running = False
        self.compare_button.setEnabled(True)
        self.set_status("Compare page reset.")

    def _invalidate_result(self, status: str = "") -> None:
        self._compare_generation += 1
        self.result = None
        self.result_text.clear()
        self.viewer_a.set_faces([])
        self.viewer_b.set_faces([])
        if status:
            self.set_status(status)

    def export_result(self) -> None:
        if self.result is None:
            self.show_error("No compare result to export.")
            return
        default = str(Path(self.context.config.export_dir) / f"compare_{timestamp_for_filename()}.json")
        path = self.save_file("Export Result", default, "JSON (*.json);;Markdown (*.md)")
        if not path:
            return
        if path.lower().endswith(".md"):
            text = "\n".join(
                [
                    "# InsightFace 1:1 Compare Result",
                    f"- Image A: {self.path_a}",
                    f"- Image B: {self.path_b}",
                    f"- Similarity: {self.result.similarity:.4f}",
                    f"- Threshold: {self.result.threshold:.4f}",
                    f"- Decision: {self.result.decision}",
                    f"- Notes: {'; '.join(self.result.notes)}",
                ]
            )
            export_markdown(path, text)
        else:
            export_json(path, self.result.to_json_dict())
        self.set_status(f"Result exported to {path}")
