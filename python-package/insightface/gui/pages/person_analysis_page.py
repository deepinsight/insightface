"""A single-input demonstration of detection, matching and reference updates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import QEvent, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGraphicsItem, QGraphicsRectItem, QGroupBox, QHBoxLayout,
    QGraphicsOpacityEffect, QHeaderView, QLabel, QLineEdit, QScrollArea, QSpinBox, QSplitter,
    QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..app import begin_context_activity, context_activity_count, end_context_activity
from ...app.person.config import PersonConfig
from ..core.config import save_config, validate_person_config
from ..core.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from ..core.i18n import apply_translations, tr
from ..core.links import open_insightface_url
from ..core.model_packages import inspect_person_model, person_provider_runtime_display
from ..core.person_analysis import PersonAnalysisJob, PersonAnalysisRunner, display_time
from ..widgets.image_viewer import ImageViewer
from ..widgets.upload_preview import UploadPreview
from .base import BasePage
from .license_center_page import ENTERPRISE_HELP_URL


# Only presentation copy lives here. Field membership, types and defaults come
# from the public API, so the dialog cannot carry a separate set of defaults.
PERSON_PARAMETER_COPY = {
    "face_similarity_threshold": ("Face matching threshold", "Minimum similarity needed to match a registered face."),
    "reid_similarity_threshold": ("Body matching threshold", "Minimum similarity needed to match a registered body."),
    "face_margin": ("Face match lead", "Required similarity lead over the next person's face match."),
    "reid_margin": ("Body match lead", "Required similarity lead over the next person's body match."),
    "face_min_size": ("Minimum face size (pixels)", "Smallest face width and height used for analysis."),
    "face_registration_min_size": ("Minimum reference face size (pixels)", "Smallest face width and height accepted in reference photos."),
    "face_min_score": ("Minimum face detection confidence", "Minimum detection confidence for using a face."),
    "face_det_size": ("Face detection input size (pixels)", "Square input width and height. The default is 640; enter a positive multiple of 32 to override it."),
    "body_min_size": ("Minimum body size (pixels)", "Smallest body width and height used for analysis."),
    "body_min_score": ("Minimum body detection confidence", "Minimum detection confidence for using a body."),
    "body_det_size": ("Body detection input size (pixels)", "Square input width and height. Use the model default, or enter a positive multiple of 64."),
    "face_body_margin": ("Face and body association lead", "Required lead when linking a face to a detected body."),
    "max_body_samples": ("Body samples per person", "Maximum stored body samples for each registered person."),
    "reference_capacity": ("Initial reference capacity", "Initial number of sample slots; storage grows when needed."),
    "duplicate_similarity_threshold": ("Duplicate sample threshold", "Samples at or above this similarity are treated as duplicates."),
    "cpu_threads": ("CPU threads", "Number of CPU threads used by each inference session."),
}
DETECTION_SIZE_STEPS = {"face_det_size": 32, "body_det_size": 64}


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Keep numeric precision without making every value display trailing zeros."""

    def textFromValue(self, value):
        text = super().textFromValue(value)
        if self.locale().decimalPoint() in text:
            text = text.rstrip(self.locale().zeroDigit()).rstrip(self.locale().decimalPoint())
        return text


class PersonParametersDialog(QDialog):
    """Edit a private draft; only a successful Save updates the page's settings."""

    def __init__(self, overrides, language, save, parent=None):
        super().__init__(parent)
        self.setObjectName("personParametersDialog")
        self.setWindowTitle("Advanced parameters")
        self.setMinimumWidth(520)
        self.resize(680, 660)
        self._language = language
        self._save_error = None
        self._save = save
        self.defaults = asdict(PersonConfig())
        current = asdict(PersonConfig(**validate_person_config(overrides)))
        self.controls = {}
        self.field_labels = {}
        self.help_labels = {}
        layout = QVBoxLayout(self)
        self.note = QLabel("Changes apply the next time you start analysis.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        form = QFormLayout(content)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(16)
        for field in fields(PersonConfig):
            name = field.name
            default = self.defaults[name]
            title, help_text = PERSON_PARAMETER_COPY[name]
            if type(default) is int:
                control = QSpinBox()
                control.setRange(0 if name in DETECTION_SIZE_STEPS else 1, 2147483647)
                if name in DETECTION_SIZE_STEPS:
                    control.setSingleStep(DETECTION_SIZE_STEPS[name])
            else:
                control = CompactDoubleSpinBox()
                control.setDecimals(15)
                control.setRange(0, 1)
                control.setSingleStep(.01)
            control.setObjectName("personParameter_" + name)
            control.setValue(current[name])
            control.setToolTip(help_text)
            control.setKeyboardTracking(False)
            control.setMaximumWidth(170)
            label = QLabel(title)
            label.setWordWrap(True)
            label.setBuddy(control)
            help_label = QLabel(help_text)
            help_label.setWordWrap(True)
            help_label.setProperty("role", "muted")
            details = QWidget()
            details_layout = QVBoxLayout(details)
            details_layout.setContentsMargins(0, 0, 0, 0)
            details_layout.setSpacing(4)
            details_layout.addWidget(control)
            details_layout.addWidget(help_label)
            form.addRow(label, details)
            self.controls[name] = control
            self.field_labels[name] = label
            self.help_labels[name] = help_label
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll, 1)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setProperty("role", "status")
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox()
        self.restore_button = self.buttons.addButton("Restore defaults", QDialogButtonBox.ResetRole)
        self.save_button = self.buttons.addButton("Save", QDialogButtonBox.AcceptRole)
        self.cancel_button = self.buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.restore_button.clicked.connect(self.restore_defaults)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        apply_translations(self, language)

    def restore_defaults(self):
        for name, control in self.controls.items():
            control.setValue(self.defaults[name])
        self.error_label.hide()
        self._save_error = None

    def overrides(self):
        values = {}
        for name, control in self.controls.items():
            control.interpretText()
            values[name] = control.value()
        validate_person_config(values)
        return {name: value for name, value in values.items() if value != self.defaults[name]}

    def accept(self):
        try:
            self._save(self.overrides())
        except Exception as exc:
            self._save_error = str(exc)
            self.retranslate_dynamic_content(self._language)
            self.error_label.show()
            return
        super().accept()

    def retranslate_dynamic_content(self, language=None):
        self._language = language or self._language
        for name in DETECTION_SIZE_STEPS:
            label = "Default (640)" if name == "face_det_size" else "Model default"
            self.controls[name].setSpecialValueText(tr(label, self._language))
        if self._save_error is not None:
            self.error_label.setText(tr("Could not save person settings:\n{error}", self._language).format(error=self._save_error))


class PersonPreview(ImageViewer):
    """Use Qt text so names in reference photos can contain any GUI language."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.matches = []
        self.language = "en"
        self.setMinimumSize(360, 220)

    def set_result(self, image, result, language):
        self.matches = result.matches
        self.language = language
        self.set_image(image)

    def draw_overlays(self):
        if self.pixmap_item is None:
            return
        for item in list(self.scene.items()):
            # A label owns its background; removing its parent removes both.
            if item is not self.pixmap_item and item.parentItem() is None:
                self.scene.removeItem(item)
        for match in self.matches:
            person = match.observation
            known = match.person_id is not None
            color = QColor("#10b981" if known else "#f59e0b")
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            face = person.face
            box = person.body_bbox
            if box is None:
                if face is None:
                    continue
                box = face.bbox
                pen.setStyle(Qt.DashLine)
            x1, y1, x2, y2 = map(float, box)
            self.scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)
            if face is not None and person.body_bbox is not None:
                fx1, fy1, fx2, fy2 = map(float, face.bbox)
                face_pen = QPen(color, 1, Qt.DashLine)
                face_pen.setCosmetic(True)
                self.scene.addRect(QRectF(fx1, fy1, fx2 - fx1, fy2 - fy1), face_pen)
            label = str(match.person_id) if known else tr("Unmatched", self.language)
            if known and match.matched_by in {"face", "body"} and match.similarity is not None:
                basis = tr("Face" if match.matched_by == "face" else "Body", self.language)
                label += f" · {basis} {match.similarity:.2f}"
            scale = max(.01, min(self.viewport().width() / self.image.shape[1],
                                self.viewport().height() / self.image.shape[0]))
            text = self.scene.addSimpleText(label)
            text.setToolTip(label)
            text.setBrush(QColor("#ffffff"))
            text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            rect = text.boundingRect()
            tx = max(0, min(x1, self.image.shape[1] - (rect.width() + 8) / scale))
            ty = max(0, y1 - (rect.height() + 6) / scale)
            background = QGraphicsRectItem(rect.adjusted(-4, -2, 4, 2), text)
            background.setPen(QPen(Qt.NoPen))
            background.setBrush(QColor("#172435"))
            background.setFlag(QGraphicsItem.ItemStacksBehindParent)
            text.setPos(tx, ty)
            text.setZValue(2)


class PersonAnalysisPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Person Analysis",
                         "Detect people and compare them with your reference photos.", parent)
        self.setObjectName("personAnalysisPage")
        self.runner = None
        self.worker = None
        self._running = False
        self._failed = False
        self._last_result = None

        model_card, model_layout = self.card()
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        self.model_status.setProperty("role", "muted")
        self.models_button = self.button("Choose model", self.open_models)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_label, 1)
        model_row.addWidget(self.models_button)
        model_layout.addLayout(model_row)
        model_layout.addWidget(self.model_status)
        self.content.addWidget(model_card)

        split = QSplitter(Qt.Horizontal)
        self.operation_panel = split
        self.operation_panel.setObjectName("personAnalysisOperations")
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        source_group = QGroupBox("1. Choose input")
        source_layout = QVBoxLayout(source_group)
        self.input_kind = QComboBox()
        self.input_kind.setProperty("i18nItems", True)
        for label, value in (("Local video", "video"), ("Local camera", "camera"), ("Remote camera (RTSP)", "rtsp")):
            self.input_kind.addItem(label, value)
        source_layout.addWidget(self.input_kind)
        self.video_input = UploadPreview("Video", VIDEO_EXTENSIONS,
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All Files (*)")
        self.video_input.setMinimumHeight(140)
        self.video_input.file_label.setWordWrap(True)
        self.video_input.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        source_layout.addWidget(self.video_input)
        self.camera_controls = QWidget()
        camera_form = QFormLayout(self.camera_controls)
        camera_form.setContentsMargins(0, 0, 0, 0)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 32)
        camera_form.addRow("Camera number", self.camera_index)
        source_layout.addWidget(self.camera_controls)
        self.rtsp_controls = QWidget()
        rtsp_layout = QVBoxLayout(self.rtsp_controls)
        rtsp_layout.setContentsMargins(0, 0, 0, 0)
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText("rtsp://camera-address/stream")
        self.rtsp_url.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.rtsp_url.setToolTip("Camera address is used for this run only and is not saved in settings or reports.")
        rtsp_layout.addWidget(self.rtsp_url)
        source_layout.addWidget(self.rtsp_controls)
        self.analysis_rate_controls = QWidget()
        rate_form = QFormLayout(self.analysis_rate_controls)
        rate_form.setContentsMargins(0, 0, 0, 0)
        rate_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.analysis_max_fps = CompactDoubleSpinBox()
        self.analysis_max_fps.setObjectName("personAnalysisMaxFps")
        self.analysis_max_fps.setDecimals(6)
        self.analysis_max_fps.setRange(0, max(1000, context.config.person_analysis_max_fps))
        self.analysis_max_fps.setSingleStep(1)
        self.analysis_max_fps.setSpecialValueText(self._tr("Auto"))
        self.analysis_max_fps.setValue(context.config.person_analysis_max_fps)
        self.analysis_max_fps.setKeyboardTracking(False)
        self.analysis_max_fps.setToolTip("Applies to videos and cameras. Auto processes every video frame and the latest camera frame as fast as the device allows.")
        rate_form.addRow("Analyze at most (times/second)", self.analysis_max_fps)
        self.analysis_rate_hint = QLabel("Applies to videos and cameras. Auto processes every video frame and the latest camera frame as fast as the device allows.")
        self.analysis_rate_hint.setWordWrap(True)
        self.analysis_rate_hint.setProperty("role", "muted")
        rate_form.addRow(self.analysis_rate_hint)
        source_layout.addWidget(self.analysis_rate_controls)
        self.input_hint = QLabel()
        self.input_hint.setWordWrap(True)
        self.input_hint.setProperty("role", "muted")
        source_layout.addWidget(self.input_hint)
        controls_layout.addWidget(source_group)

        references_group = QGroupBox("2. Add people (optional)")
        self.references_group = references_group
        references_layout = QVBoxLayout(references_group)
        note = QLabel("Drag photos here or click Add photos. Use one clear face per photo. Edit the name below; photos with the same name belong to one person. Photos are optional.")
        note.setWordWrap(True)
        references_layout.addWidget(note)
        self.references_table = QTableWidget(0, 2)
        self.references_table.setHorizontalHeaderLabels(["Name", "Reference photo"])
        self.references_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.references_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.references_table.setMaximumHeight(120)
        self.references_table.verticalHeader().hide()
        self.add_references_button = self.button("Add photos", self.add_references)
        self.remove_references_button = self.button("Remove selected", self.remove_references)
        references_layout.addWidget(self.row(self.add_references_button, self.remove_references_button))
        references_layout.addWidget(self.references_table)
        self.auto_update = QCheckBox("Automatically add body references")
        self.auto_update.setChecked(True)
        references_layout.addWidget(self.auto_update)
        update_note = QLabel("A reliable face match can add body references for this run. Unmatched people are not enrolled automatically.")
        update_note.setWordWrap(True)
        update_note.setProperty("role", "muted")
        references_layout.addWidget(update_note)
        # Cover the whole reference area, including the table's viewport and
        # buttons, so child widgets do not swallow external photo drops.
        self._reference_drop_targets = {references_group, *references_group.findChildren(QWidget)}
        for widget in self._reference_drop_targets:
            widget.setAcceptDrops(True)
            widget.installEventFilter(self)
        controls_layout.addWidget(references_group)
        self.advanced_button = self.button("Advanced parameters", self.open_advanced_parameters)
        self.advanced_button.setObjectName("personAdvancedParameters")
        self.advanced_button.setToolTip("Changes apply the next time you start analysis.")
        self.start_button = self.button("Start analysis", self.start)
        self.start_button.setObjectName("personAnalysisStart")
        self.stop_button = self.button("Stop", self.stop, enabled=False)
        controls_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(310)
        input_panel = QWidget()
        input_panel_layout = QVBoxLayout(input_panel)
        input_panel_layout.setContentsMargins(0, 0, 0, 0)
        input_panel_layout.addWidget(scroll, 1)
        actions = QWidget()
        actions_layout = QVBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(self.advanced_button)
        actions_layout.addWidget(self.row(self.start_button, self.stop_button))
        input_panel_layout.addWidget(actions)
        split.addWidget(input_panel)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        self.preview = PersonPreview()
        results_layout.addWidget(self.preview, 3)
        self.stats_label = QLabel("Preview appears after analysis starts.")
        self.stats_label.setWordWrap(True)
        results_layout.addWidget(self.stats_label)
        legend = QLabel("Green: matched reference · Amber: unmatched · Dashed box: face")
        legend.setProperty("role", "muted")
        legend.setWordWrap(True)
        results_layout.addWidget(legend)
        result_note = QLabel("This demo shows the current frame. References are kept in memory for this run; no history database is created.")
        result_note.setProperty("role", "muted")
        result_note.setWordWrap(True)
        results_layout.addWidget(result_note)
        split.addWidget(results)
        split.setSizes([330, 720])
        split.setStretchFactor(1, 1)
        self.content.addWidget(split, 1)
        # Dim the scroll contents, not QScrollArea's opaque viewport: applying
        # opacity to the viewport ancestor can paint a dark rectangle on macOS.
        self.operation_opacities = []
        for panel in (controls, actions, results):
            opacity = QGraphicsOpacityEffect(panel)
            opacity.setOpacity(1.0)
            panel.setGraphicsEffect(opacity)
            self.operation_opacities.append(opacity)

        commercial, commercial_layout = self.card()
        commercial_text = QLabel("Ready to use this in your product? Contact InsightFace for commercial model licensing and SDK integration.")
        commercial_text.setWordWrap(True)
        self.commercial_button = self.button("Get commercial license", self.open_commercial)
        commercial_row = QHBoxLayout()
        commercial_row.addWidget(commercial_text, 1)
        commercial_row.addWidget(self.commercial_button)
        commercial_layout.addLayout(commercial_row)
        self.content.addWidget(commercial)

        self._editable = [source_group, references_group, self.models_button, self.advanced_button]
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll)
        self.input_kind.currentIndexChanged.connect(self._source_changed)
        self.video_input.pathChanged.connect(lambda _: self.refresh())
        self.rtsp_url.textChanged.connect(lambda _: self.refresh())
        self.analysis_max_fps.editingFinished.connect(self._save_analysis_rate)
        self._source_changed()

    def _tr(self, text):
        return tr(text, self.context.config.ui_language)

    def _source_changed(self):
        kind = self.input_kind.currentData()
        self.video_input.setVisible(kind == "video")
        self.camera_controls.setVisible(kind == "camera")
        self.rtsp_controls.setVisible(kind == "rtsp")
        self.input_hint.setText(self._tr(
            "Auto processes every video frame in order. A limit samples by video time. Analysis may run faster or slower than playback."
            if kind == "video" else
            "Camera frames are read continuously. Analysis uses the latest frame and skips older pending frames."
        ))
        self.refresh()

    def add_references(self):
        self.add_reference_paths(self.choose_files("Add reference photos"))

    def eventFilter(self, watched, event):
        if (watched in self._reference_drop_targets and
                event.type() in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop)):
            paths = []
            if not self._running and self.references_group.isEnabled():
                for url in event.mimeData().urls():
                    if url.isLocalFile():
                        path = Path(url.toLocalFile())
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                            paths.append(str(path))
            if paths:
                if event.type() == QEvent.Drop:
                    self.add_reference_paths(paths)
                event.setDropAction(Qt.CopyAction)
                event.accept()
            else:
                event.ignore()
            return True
        return super().eventFilter(watched, event)

    def add_reference_paths(self, paths):
        if self._running:
            return
        existing = {self.references_table.item(row, 1).data(Qt.UserRole)
                    for row in range(self.references_table.rowCount())}
        for path in paths:
            path = str(Path(path).expanduser().resolve())
            if path in existing:
                continue
            row = self.references_table.rowCount()
            self.references_table.insertRow(row)
            self.references_table.setItem(row, 0, QTableWidgetItem(Path(path).stem))
            item = QTableWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.references_table.setItem(row, 1, item)
            existing.add(path)

    def remove_references(self):
        for row in sorted({item.row() for item in self.references_table.selectedItems()}, reverse=True):
            self.references_table.removeRow(row)

    def open_models(self):
        main = self.window()
        if hasattr(main, "open_model_manager"):
            main.open_model_manager()

    def _save_person_parameters(self, overrides):
        cfg = self.context.config
        draft = deepcopy(cfg)
        draft.person_config = validate_person_config(overrides)
        save_config(draft)
        cfg.person_config = draft.person_config

    def open_advanced_parameters(self):
        if self._running or not self.advanced_button.isEnabled():
            return
        try:
            dialog = PersonParametersDialog(
                self.context.config.person_config, self.context.config.ui_language,
                self._save_person_parameters, self,
            )
        except Exception as exc:
            self.show_error(self._tr("Could not load person settings:\n{error}").format(error=exc))
            return
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _save_analysis_rate(self):
        cfg = self.context.config
        if self._running or self.analysis_max_fps.value() == cfg.person_analysis_max_fps:
            return
        draft = deepcopy(cfg)
        draft.person_analysis_max_fps = self.analysis_max_fps.value()
        try:
            save_config(draft)
        except Exception as exc:
            self.analysis_max_fps.setValue(cfg.person_analysis_max_fps)
            self.show_error(self._tr("Could not save person settings:\n{error}").format(error=exc))
            return
        cfg.person_analysis_max_fps = draft.person_analysis_max_fps

    def open_commercial(self):
        open_insightface_url(ENTERPRISE_HELP_URL, content="person_analysis_commercial")

    def refresh(self):
        cfg = self.context.config
        status = inspect_person_model(cfg.model_name, cfg.model_root)
        unsupported = status.state == "unsupported" and not self._running
        self.operation_panel.setEnabled(not unsupported)
        self.advanced_button.setEnabled(not self._running and not unsupported)
        for opacity in self.operation_opacities:
            opacity.setOpacity(.46 if unsupported else 1.0)
        provider, _ = person_provider_runtime_display(cfg.provider, cfg.ui_language)
        self.model_label.setText(f"{self._tr('Model')}: {cfg.model_name}  ·  {provider}")
        messages = {
            "unsupported": "Choose cheetah_s (fast) or cheetah_l (larger) in Models to use Person Analysis.",
            "missing": "Model is not installed. Start will try downloading it; you can also place the package in the folder shown below.",
            "ready": "Local model is ready. Analysis uses the shared model and device settings.",
        }
        self.model_status.setText(self._tr(messages.get(status.state, status.message)) + "\n" + str(status.package_path))
        self.model_status.setToolTip(status.message)
        kind = self.input_kind.currentData()
        has_source = kind == "camera" or (kind == "video" and bool(self.video_input.path())) or (kind == "rtsp" and bool(self.rtsp_url.text().strip()))
        self.start_button.setEnabled(not self._running and status.can_start and has_source)

    def _job(self):
        cfg = self.context.config
        kind = self.input_kind.currentData()
        source = self.video_input.path() if kind == "video" else self.camera_index.value() if kind == "camera" else self.rtsp_url.text().strip()
        if kind == "video" and not Path(source).is_file():
            raise ValueError(self._tr("Choose an existing local video."))
        if kind == "rtsp":
            try:
                parts = urlsplit(source)
                valid = parts.scheme in {"rtsp", "rtsps"} and bool(parts.hostname)
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(self._tr("Enter a valid RTSP camera address."))
        refs = []
        for row in range(self.references_table.rowCount()):
            name = self.references_table.item(row, 0).text().strip()
            if not name:
                raise ValueError(self._tr("Enter a name for every reference photo."))
            refs.append((name, self.references_table.item(row, 1).data(Qt.UserRole)))
        return PersonAnalysisJob(
            model_name=cfg.model_name, model_root=Path(cfg.model_root).expanduser(),
            provider_choice=cfg.provider, source_kind=kind, source=source,
            references=tuple(refs),
            cache_dir=Path(cfg.cache_dir),
            auto_update=self.auto_update.isChecked(),
            person_config=cfg.person_config,
            analysis_max_fps=self.analysis_max_fps.value(),
        )

    def start(self):
        if self._running:
            return
        if any(context_activity_count(self.context, key) for key in (
                "model_downloads_in_progress", "privateframe_jobs_in_progress", "person_analysis_jobs_in_progress")):
            self.show_error(self._tr("Wait for the current model download or video task to finish."))
            return
        try:
            status = inspect_person_model(self.context.config.model_name, self.context.config.model_root)
            if not status.can_start:
                raise ValueError(status.message)
            job = self._job()
            self.runner = PersonAnalysisRunner(job)
        except Exception as exc:
            self.show_error(str(exc))
            return
        self._failed = False
        self._last_result = None
        self.preview.matches = []
        self.preview.set_image(None)
        self._running = True
        begin_context_activity(self.context, "person_analysis_jobs_in_progress")
        for widget in self._editable:
            widget.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stats_label.setText(self._tr("Loading model and reference photos…"))
        self.timer.start()
        try:
            self.worker = self.run_task(
                "Person Analysis", self.runner.run, self._done, show_dialog=False,
                on_progress=lambda _current, _total, text: self.set_status(self._tr(text)),
                on_error=self._error, on_finished=self._finished,
            )
        except Exception as exc:
            self._error(str(exc))
            self._finished()

    def stop(self):
        if self.runner is not None and self._running:
            self.runner.request_stop()
            if self.worker is not None:
                self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.set_status("Stopping input… Waiting for the current read or inference to finish.")

    def _poll(self):
        if self.runner is None:
            return
        preview = self.runner.take_preview()
        if preview is not None:
            frame, result = preview
            self._last_result = result
            self.preview.set_result(frame, result, self.context.config.ui_language)
            self._update_stats()

    def _update_stats(self):
        result = self._last_result
        if result is None:
            return
        identified = sum(match.person_id is not None for match in result.matches)
        self.stats_label.setText(self._tr("Analyzed frames: {frame} · Time: {time} · Detected: {count} · Matched: {known}").format(
            frame=result.frame_index, time=display_time(result.timestamp, result.time_basis),
            count=len(result.matches), known=identified))

    def _done(self, summary):
        self._poll()
        self.set_status(self._tr("Stopped. Processed {count} frames.").format(count=summary.get("frames_processed", 0))
                        if summary.get("stopped") else self._tr("Finished. Processed {count} frames.").format(count=summary.get("frames_processed", 0)))

    def _error(self, message):
        self._failed = True
        self.show_error(message)

    def _finished(self):
        self._poll()
        self.timer.stop()
        self._running = False
        self.worker = None
        end_context_activity(self.context, "person_analysis_jobs_in_progress")
        for widget in self._editable:
            widget.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.refresh()

    def retranslate_dynamic_content(self, _language=None):
        self.analysis_max_fps.setSpecialValueText(self._tr("Auto"))
        self._source_changed()
        self._update_stats()
        self.preview.language = self.context.config.ui_language
        self.preview.draw_overlays()

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
