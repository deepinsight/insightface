"""PrivateFrame video redaction workspace."""

from __future__ import annotations

import math
import logging
import threading
from contextlib import contextmanager
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
from PySide6.QtCore import QEvent, QStandardPaths, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...app.privateframe.base_config import DEFAULT_CONFIG_PATH, read_default_config
from ...app.privateframe.output_paths import default_output_paths
from ...app.privateframe.recognition import scan_gallery, resolve_unknown_action
from ...app.privateframe.pipeline import (
    analyze_streaming_pipeline,
    run_streaming_pipeline,
)
from ...app.privateframe.video import probe_video
from ..app import (
    begin_context_activity,
    context_activity_count,
    end_context_activity,
)
from ..core.face_engine import provider_runtime_display
from ..core.i18n import tr
from ..core.model_packages import (
    PRIVATEFRAME_MODEL_PACKAGES,
    PrivateFrameModelStatus,
    inspect_privateframe_model,
)
from ..core.tooltips import set_button_tooltip
from ..core.video import read_video_thumbnail
from ..widgets.upload_preview import UploadPreview
from .base import BasePage


_MODEL_PACKAGES = PRIVATEFRAME_MODEL_PACKAGES
_REDACTION_METHODS = {"gaussian", "mosaic"}
_OUTPUT_MODES = {"json_and_video", "json_only"}
_BETWEEN_SCAN_MODES = {"auto", "interpolate", "visual"}
_RECOGNITION_MODES = {"all", "exempt", "blur_only"}
_RECOGNITION_PROFILES = {"fast", "balanced", "accurate"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
_VIDEO_PREVIEW_SIZE = 120
_UNAVAILABLE_CONTENT_OPACITY = 0.46
_PROVIDER_NAMES = {
    "auto": "auto",
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "CPUExecutionProvider": "CPUExecutionProvider",
    "CUDAExecutionProvider": "CUDAExecutionProvider",
    "CoreMLExecutionProvider": "CoreMLExecutionProvider",
}


def _default_privateframe_output_directory() -> Path:
    """Return the platform's user-visible video directory without creating it."""

    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.MoviesLocation
    ).strip()
    if location:
        return Path(location).expanduser()
    folder_name = "Movies" if sys.platform == "darwin" else "Videos"
    return Path.home() / folder_name


def _video_dialog_filter(language: str | None) -> str:
    return (
        f"{tr('Videos', language)} (*.mp4 *.mov *.m4v *.mkv *.avi *.webm);;"
        f"{tr('All Files', language)} (*)"
    )


@dataclass(frozen=True)
class PrivateFrameJob:
    input_path: Path
    output_dir: Path
    config_path: Path
    workdir: Path
    result_path: Path
    redacted_path: Path | None
    output_mode: str
    model_name: str
    model_root: Path
    provider_choice: str
    config_overrides: dict[str, object]


@dataclass(frozen=True)
class VideoPreviewData:
    """Small GUI-safe video preview and the metadata shown beside it."""

    path: Path
    image: np.ndarray
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    file_size: int
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool | None


def _letterbox_square(image: np.ndarray, size: int = _VIDEO_PREVIEW_SIZE) -> np.ndarray:
    """Resize a BGR frame into a square canvas while preserving its aspect ratio."""

    if size <= 0:
        raise ValueError("Preview size must be positive")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Video preview must be a BGR color image")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Video preview has invalid dimensions")

    import cv2

    scale = min(size / width, size / height)
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image[..., :3],
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _read_video_preview(path: str | Path) -> VideoPreviewData:
    """Read one compact preview and inexpensive stream metadata off the GUI thread."""

    source = Path(path).expanduser().resolve()
    metadata = probe_video(source)
    frame = read_video_thumbnail(source)
    if frame is None:
        raise RuntimeError(f"Could not read a preview frame: {source}")

    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool | None = None
    try:
        container = av.open(str(source))
        try:
            video_stream = next(iter(container.streams.video), None)
            audio_stream = next(iter(container.streams.audio), None)
            if video_stream is not None:
                video_codec = str(video_stream.codec_context.name or "") or None
            has_audio = audio_stream is not None
            if audio_stream is not None:
                audio_codec = str(audio_stream.codec_context.name or "") or None
        finally:
            container.close()
    except Exception:
        # Codec labels are supplementary. A valid OpenCV preview remains useful
        # even when the local PyAV build cannot inspect this container.
        pass

    return VideoPreviewData(
        path=source,
        image=_letterbox_square(frame),
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        duration=metadata.duration,
        file_size=source.stat().st_size,
        video_codec=video_codec,
        audio_codec=audio_codec,
        has_audio=has_audio,
    )


def _format_video_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "—"
    total_centiseconds = int(round(seconds * 100.0))
    hours, remainder = divmod(total_centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    second, centiseconds = divmod(remainder, 100)
    if hours:
        return f"{hours:d}:{minutes:02d}:{second:02d}.{centiseconds:02d}"
    return f"{minutes:02d}:{second:02d}.{centiseconds:02d}"


def _format_file_size(byte_count: int) -> str:
    value = float(max(0, byte_count))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            precision = 0 if unit == "B" else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def build_privateframe_job(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model_package: str,
    max_analysis_fps: int | float | None = None,
    redaction_method: str | None = None,
    runtime_provider: str,
    model_root: str | Path | None = None,
    preserve_aac_audio: bool | None = None,
    output_mode: str = "json_and_video",
    between_scan_frames: str | None = None,
    box_scale: float | None = None,
    video_preset: str | None = None,
    video_crf: int | None = None,
    recognition_mode: str | None = None,
    recognition_reference_dir: str | Path | None = None,
    recognition_profile: str | None = None,
    recognition_similarity_threshold: float | None = None,
    base_defaults: dict[str, Any] | None = None,
) -> PrivateFrameJob:
    """Validate GUI selections and materialize deterministic output paths."""

    defaults = read_default_config(DEFAULT_CONFIG_PATH) if base_defaults is None else base_defaults
    scan_defaults = defaults["scan"]
    redaction_defaults = defaults["render"]["redaction"]
    video_defaults = defaults["render"]["video_output"]
    recognition_defaults = defaults["recognition"]
    max_analysis_fps = scan_defaults["max_analysis_fps"] if max_analysis_fps is None else max_analysis_fps
    redaction_method = redaction_defaults["method"] if redaction_method is None else redaction_method
    model_root = defaults["models"]["root"] if model_root is None else model_root
    between_scan_frames = defaults["tracking"]["between_scan_frames"] if between_scan_frames is None else between_scan_frames
    box_scale = redaction_defaults["box_scale"] if box_scale is None else box_scale
    video_preset = video_defaults["preset"] if video_preset is None else video_preset
    recognition_mode = recognition_defaults["mode"] if recognition_mode is None else recognition_mode
    recognition_profile = recognition_defaults["profile"] if recognition_profile is None else recognition_profile
    recognition_similarity_threshold = recognition_defaults["similarity_threshold"] if recognition_similarity_threshold is None else recognition_similarity_threshold
    audio_mode = video_defaults["audio"]["redacted"] if preserve_aac_audio is None else "aac" if preserve_aac_audio else "none"
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if model_package not in _MODEL_PACKAGES:
        raise ValueError(f"Unsupported PrivateFrame model package: {model_package}")
    if (
        isinstance(max_analysis_fps, bool)
        or not isinstance(max_analysis_fps, (int, float))
        or not math.isfinite(float(max_analysis_fps))
        or float(max_analysis_fps) <= 0
    ):
        raise ValueError(f"Unsupported maximum analysis FPS: {max_analysis_fps}")
    normalized_max_analysis_fps = float(max_analysis_fps)
    if redaction_method not in _REDACTION_METHODS:
        raise ValueError(f"Unsupported redaction method: {redaction_method}")
    if output_mode not in _OUTPUT_MODES:
        raise ValueError(f"Unsupported PrivateFrame output mode: {output_mode}")
    if between_scan_frames not in _BETWEEN_SCAN_MODES:
        raise ValueError(
            f"Unsupported between-scan processing mode: {between_scan_frames}"
        )
    try:
        normalized_box_scale = float(box_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported redaction box scale: {box_scale}") from exc
    if isinstance(box_scale, bool) or not math.isfinite(normalized_box_scale) or not 0.1 <= normalized_box_scale <= 4.0:
        raise ValueError(f"Unsupported redaction box scale: {box_scale}")
    if video_preset is not None and not isinstance(video_preset, str):
        raise ValueError(f"Unsupported video preset: {video_preset}")
    if video_crf is not None and (type(video_crf) is not int or not 0 <= video_crf <= 51):
        raise ValueError(f"Unsupported video CRF quality: {video_crf}")
    if recognition_mode not in _RECOGNITION_MODES:
        raise ValueError(f"Unsupported recognition policy: {recognition_mode}")
    if recognition_profile not in _RECOGNITION_PROFILES:
        raise ValueError(f"Unsupported recognition profile: {recognition_profile}")

    if (
        isinstance(recognition_similarity_threshold, bool)
        or not isinstance(recognition_similarity_threshold, (int, float))
        or not math.isfinite(float(recognition_similarity_threshold))
        or not 0.0 <= float(recognition_similarity_threshold) <= 1.0
    ):
        raise ValueError("Match threshold must be between 0 and 1.")

    provider = _PROVIDER_NAMES.get(
        runtime_provider, _PROVIDER_NAMES.get(runtime_provider.lower())
    )
    if provider is None:
        raise ValueError(f"Unsupported PrivateFrame provider: {runtime_provider}")

    config_path = DEFAULT_CONFIG_PATH.resolve()
    resolved_model_root = Path(model_root).expanduser().resolve()
    paths = default_output_paths(source, destination)
    render_video = output_mode == "json_and_video"
    overrides: dict[str, object] = {
        "models.name": model_package,
        "models.root": str(resolved_model_root),
        "scan.max_analysis_fps": normalized_max_analysis_fps,
        "runtime.provider": provider,
        "render.redaction.method": redaction_method,
        "render.redaction.box_scale": normalized_box_scale,
        "render.video_output.preset": video_preset,
        "render.video_output.audio.redacted": audio_mode,
        "recognition.mode": recognition_mode,
        "recognition.unknown_action": recognition_defaults["unknown_action"],
    }
    overrides["render.video_output.rate_control"] = (
        dict(video_defaults["rate_control"]) if video_crf is None
        else {"mode": "crf", "quality": video_crf}
    )
    if between_scan_frames != "auto":
        overrides["tracking.between_scan_frames"] = between_scan_frames

    # All mode never resolves or reads any reference photo argument.
    if recognition_mode != "all":
        if recognition_reference_dir is None:
            reference = recognition_defaults["reference_dir"]
            if reference is not None:
                reference_path = Path(str(reference)).expanduser()
                if not reference_path.is_absolute():
                    reference_path = DEFAULT_CONFIG_PATH.parent / reference_path
                recognition_reference_dir = reference_path
        reference_dir, _images = _scan_reference_folder(recognition_reference_dir)
        overrides.update({
            "recognition.reference_dir": str(reference_dir),
            "recognition.profile": recognition_profile,
            "recognition.similarity_threshold": float(recognition_similarity_threshold),
        })
    return PrivateFrameJob(
        input_path=source,
        output_dir=destination,
        config_path=config_path,
        workdir=paths.workdir,
        result_path=paths.result_json,
        redacted_path=paths.result_video if render_video else None,
        output_mode=output_mode,
        model_name=model_package,
        model_root=resolved_model_root,
        provider_choice=str(runtime_provider),
        config_overrides=overrides,
    )


def _scan_reference_folder(value: str | Path | None):
    text = str(value or "").strip()
    if not text:
        raise ValueError("Select a reference photo folder.")
    unresolved = Path(text).expanduser()
    if unresolved.is_symlink():
        raise ValueError("The reference photo folder must not be a symlink.")
    root = unresolved.resolve()
    if not root.is_dir():
        raise ValueError("The reference photo folder does not exist.")
    scanned = scan_gallery(root)
    if not scanned.images:
        raise ValueError("No supported reference photos were found in this folder.")
    return root, scanned.images


_REFERENCE_REJECTION_LABELS = {
    "duplicate_content": "Duplicate photo",
    "unreadable_image": "Image could not be read",
    "no_face": "No face detected",
    "invalid_face_box": "Invalid face box",
    "missing_landmarks": "Facial landmarks unavailable",
    "invalid_landmarks": "Invalid facial landmarks",
    "invalid_confidence": "Invalid face detection confidence",
    "invalid_geometry": "Face could not be aligned",
    "low_quality": "Largest face is not clear enough for matching",
}
_REFERENCE_LOG_LOCK = threading.RLock()
_REFERENCE_LOG_USERS = 0
_REFERENCE_LOG_ORIGINAL_LEVEL = logging.NOTSET


class _ReferenceLogHandler(logging.Handler):
    """Forward only reference import records emitted by this worker thread."""

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        super().__init__(logging.INFO)
        self.thread_id = threading.get_ident()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        args = record.args
        if not isinstance(args, tuple):
            return
        if record.msg == "Reference photo %s was not used: %s":
            self.callback({"kind": "skip", "file": args[0], "reason": args[1]})
        elif str(record.msg).startswith("Reference photo %s: detected %d faces;"):
            self.callback({"kind": "largest", "file": args[0], "count": args[1]})
        elif record.msg == "Reference photos: read %d, used %d, skipped %d":
            self.callback({"kind": "summary", "total": args[0], "used": args[1], "skipped": args[2]})


@contextmanager
def _forward_reference_logs(callback):
    global _REFERENCE_LOG_USERS, _REFERENCE_LOG_ORIGINAL_LEVEL
    logger = logging.getLogger("insightface.app.privateframe.recognition")
    handler = _ReferenceLogHandler(callback)
    with _REFERENCE_LOG_LOCK:
        if not _REFERENCE_LOG_USERS:
            _REFERENCE_LOG_ORIGINAL_LEVEL = logger.level
            logger.setLevel(min(logger.getEffectiveLevel(), logging.INFO))
        _REFERENCE_LOG_USERS += 1
        logger.addHandler(handler)
    try:
        yield
    finally:
        with _REFERENCE_LOG_LOCK:
            logger.removeHandler(handler)
            handler.close()
            _REFERENCE_LOG_USERS -= 1
            if not _REFERENCE_LOG_USERS:
                logger.setLevel(_REFERENCE_LOG_ORIGINAL_LEVEL)


def _reference_log_text(event: dict[str, Any], language: str | None) -> str:
    kind = event.get("kind")
    if kind == "summary":
        return tr("Reference photos: {used} used, {skipped} skipped ({total} total).", language).format(**event)
    if kind == "largest":
        return tr("{file}: detected {count} faces; using only the largest face.", language).format(**event)
    reason = tr(_REFERENCE_REJECTION_LABELS.get(str(event.get("reason")), str(event.get("reason", ""))), language)
    return tr("{file}: photo was not used ({reason}).", language).format(file=event.get("file", ""), reason=reason)


def _privateframe_error_parts(message: str) -> tuple[str, dict[str, str]]:
    prefix = "Reference photo "
    for suffix, source in (
        (": face detector inference failed", "Reference photo {file}: face detection failed."),
        (": recognizer inference failed", "Reference photo {file}: face matching failed."),
    ):
        if message.startswith(prefix) and message.endswith(suffix):
            return source, {"file": message[len(prefix):-len(suffix)]}
    return _privateframe_error_source(message), {}


def _privateframe_error_source(message: str) -> str:
    if message.startswith("No usable faces were found in the reference photos"):
        return "No usable faces were found in the reference photos. Choose clearer photos and try again."
    return message


def run_privateframe_job(
    job: PrivateFrameJob,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    reference_log: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    logs: list[dict[str, Any]] = []

    def receive(event):
        logs.append(dict(event))
        if reference_log is not None:
            reference_log(event)

    with _forward_reference_logs(receive):
        result = _run_privateframe_job(job, progress=progress, is_cancelled=is_cancelled)
    if logs:
        result["gui"]["reference_logs"] = logs
    return result


def _run_privateframe_job(
    job: PrivateFrameJob,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the in-process PrivateFrame API from a GUI worker thread."""

    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")
    overrides = dict(job.config_overrides)
    if job.output_mode == "json_only":
        analysis = analyze_streaming_pipeline(
            config_path=job.config_path,
            input_path=job.input_path,
            workdir=job.workdir,
            result_path=job.result_path,
            config_overrides=overrides,
            config_override_root=job.config_path.parent,
            progress=progress,
            is_cancelled=is_cancelled,
        )
        return {
            "analysis": analysis,
            "render": {},
            "gui": {
                "output_mode": job.output_mode,
                "source_audio_codec": None,
                "audio_output_mode": "none",
                "audio_preference": str(overrides["render.video_output.audio.redacted"]),
            },
        }
    requested_audio_mode = str(overrides["render.video_output.audio.redacted"])
    source_audio_codec = _source_audio_codec(job.input_path)
    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")
    if requested_audio_mode == "aac" and source_audio_codec not in {None, "aac"}:
        # The current PyAV writer can remux AAC but does not yet transcode
        # arbitrary source audio. Drop incompatible audio before the expensive
        # analysis/render pass instead of failing after the full video encode.
        overrides["render.video_output.audio.redacted"] = "none"
    result = run_streaming_pipeline(
        config_path=job.config_path,
        input_path=job.input_path,
        debug_path=None,
        redacted_path=job.redacted_path,
        workdir=job.workdir,
        result_path=job.result_path,
        config_overrides=overrides,
        config_override_root=job.config_path.parent,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    result["gui"] = {
        "output_mode": job.output_mode,
        "source_audio_codec": source_audio_codec,
        "audio_output_mode": overrides["render.video_output.audio.redacted"],
    }
    return result


def _source_audio_codec(source: Path) -> str | None:
    container = av.open(str(source))
    try:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            return None
        return str(stream.codec_context.name)
    finally:
        container.close()


def _pipeline_total_seconds(
    analysis: dict[str, Any],
    render: dict[str, Any],
    *,
    render_video: bool,
) -> float | None:
    """Return the complete measured pipeline time available to the GUI.

    The analysis ``total_seconds`` value includes result-artifact writing.  An
    older/custom result may expose only ``analysis_seconds``, which remains a
    useful fallback.  UI setup and worker scheduling are intentionally outside
    this pipeline timing contract.
    """

    timings = analysis.get("timings") or {}
    value = timings.get("total_seconds", timings.get("analysis_seconds"))
    try:
        total = float(value)
        if render_video:
            total += float(render["seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(total) or total < 0.0:
        return None
    return total


def _localized_model_status(
    status: PrivateFrameModelStatus,
    language: str | None,
) -> str:
    """Translate the stable part of a model status while preserving paths/errors."""

    if status.state == "missing":
        return tr(
            "{model} is not installed under {path}. It will be downloaded there on first use.",
            language,
        ).format(model=status.model_name, path=status.model_root / "models")
    if status.state == "ready":
        return tr("Ready: {model} from {path}.", language).format(
            model=status.model_name,
            path=status.package_path,
        )
    invalid_prefix = f"PrivateFrame cannot use {status.model_name}: "
    if status.state == "invalid" and status.message.startswith(invalid_prefix):
        return tr("PrivateFrame cannot use {model}: {reason}", language).format(
            model=status.model_name,
            reason=_localized_model_reason(
                status.message[len(invalid_prefix) :],
                language,
            ),
        )
    return tr(status.message, language)


def _localized_model_reason(reason: str, language: str | None) -> str:
    """Translate stable validation prose while retaining paths and hashes."""

    if reason == "the model root is empty":
        return tr("The model root is empty.", language)
    prefixes = (
        (
            "the model root is not a directory: ",
            "The model root is not a directory: {value}",
        ),
        (
            "the models path is not a directory: ",
            "The models path is not a directory: {value}",
        ),
        (
            "the package path is not a directory: ",
            "The package path is not a directory: {value}",
        ),
        (
            "the V2 manifest is missing: ",
            "The V2 manifest is missing: {value}",
        ),
        (
            "manifest model_id is ",
            "The manifest model_id is {value}",
        ),
        (
            "missing required task(s): ",
            "Missing required task(s): {value}",
        ),
        (
            "missing model file(s): ",
            "Missing model file(s): {value}",
        ),
    )
    for prefix, source in prefixes:
        if reason.startswith(prefix):
            return tr(source, language).format(value=reason[len(prefix) :])
    sha_prefix = "model SHA-256 mismatch for "
    if (
        reason.startswith(sha_prefix)
        and ": expected " in reason
        and ", got " in reason
    ):
        path_and_expected, actual = reason[len(sha_prefix) :].rsplit(", got ", 1)
        path, expected = path_and_expected.rsplit(": expected ", 1)
        return tr(
            "Model SHA-256 mismatch for {path}: expected {expected}, got {actual}",
            language,
        ).format(path=path, expected=expected, actual=actual)
    return tr(reason, language)


class PrivateFramePage(BasePage):
    reference_log_received = Signal(object)

    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "PrivateFrame Video Privacy",
            "Blur faces or add mosaic to a video before sharing it. Choose everyone or use reference photos to select people. Your video stays on this computer.",
            parent,
        )
        self._base_defaults = read_default_config(DEFAULT_CONFIG_PATH)
        self.setObjectName("privateFramePage")
        self.root_layout.setSpacing(6)
        self._configured_combo_entries: list[tuple[QComboBox, int, str, dict[str, str]]] = []
        self._worker = None
        self._running = False
        self._privateframe_activity_registered = False
        self._cancel_requested = False
        self._last_job: PrivateFrameJob | None = None
        self._video_preview_generation = 0
        self._video_preview_worker = None
        self._video_preview_state = "empty"
        self._video_preview_path: Path | None = None
        self._video_preview_data: VideoPreviewData | None = None
        self._stage_source = "Ready"
        self._stage_values: dict[str, object] = {}
        self._progress_format_source: str | None = "Ready"
        self._reference_status_source = "Select a reference photo folder."
        self._reference_status_values: dict[str, object] = {}
        self._reference_log_entries: list[dict[str, Any]] = []
        self.reference_log_received.connect(self._reference_log_received)
        self._summary_state = "empty"
        self._summary_result: dict[str, Any] | None = None
        self._summary_error = ""
        self._summary_error_values: dict[str, str] = {}
        self._resolved_provider = ""
        self._primary_options_two_columns: bool | None = None

        self.model_requirement_banner = QFrame()
        self.model_requirement_banner.setObjectName(
            "privateFrameCompatibilityBanner"
        )
        requirement_layout = QHBoxLayout(self.model_requirement_banner)
        requirement_layout.setContentsMargins(14, 12, 14, 12)
        requirement_layout.setSpacing(14)
        requirement_copy = QVBoxLayout()
        requirement_copy.setContentsMargins(0, 0, 0, 0)
        requirement_copy.setSpacing(4)
        self.model_requirement_title = QLabel(
            "PrivateFrame requires a Raccoon model."
        )
        self.model_requirement_title.setObjectName(
            "privateFrameCompatibilityTitle"
        )
        self.model_requirement_message = QLabel()
        self.model_requirement_message.setObjectName(
            "privateFrameCompatibilityMessage"
        )
        self.model_requirement_message.setWordWrap(True)
        requirement_model_row = QHBoxLayout()
        requirement_model_row.setContentsMargins(0, 0, 0, 0)
        requirement_model_row.setSpacing(6)
        self.model_requirement_current_model_caption = QLabel(
            "Current global model"
        )
        self.model_requirement_current_model_caption.setObjectName(
            "privateFrameCompatibilityCurrentModelCaption"
        )
        self.model_requirement_current_model_value = QLabel()
        self.model_requirement_current_model_value.setObjectName(
            "privateFrameCompatibilityCurrentModelValue"
        )
        self.model_requirement_current_model_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        requirement_model_row.addWidget(
            self.model_requirement_current_model_caption
        )
        requirement_model_row.addWidget(self.model_requirement_current_model_value)
        requirement_model_row.addStretch(1)
        requirement_copy.addWidget(self.model_requirement_title)
        requirement_copy.addWidget(self.model_requirement_message)
        requirement_copy.addLayout(requirement_model_row)
        requirement_layout.addLayout(requirement_copy, 1)
        self.model_requirement_open_models_button = QPushButton("Open Models")
        self.model_requirement_open_models_button.setObjectName(
            "privateFrameCompatibilityOpenModelsButton"
        )
        self.model_requirement_open_models_button.clicked.connect(self._open_models)
        set_button_tooltip(self.model_requirement_open_models_button)
        requirement_layout.addWidget(
            self.model_requirement_open_models_button,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self.model_requirement_banner.hide()
        self.content.addWidget(self.model_requirement_banner)

        self.operation_panel = QWidget()
        self.operation_panel.setObjectName("privateFrameOperationPanel")
        self.operation_layout = QVBoxLayout(self.operation_panel)
        self.operation_layout.setContentsMargins(0, 0, 0, 0)
        self.operation_layout.setSpacing(8)
        self.operation_opacity = QGraphicsOpacityEffect(self.operation_panel)
        self.operation_opacity.setOpacity(1.0)
        self.operation_panel.setGraphicsEffect(self.operation_opacity)
        self.operation_scroll = QScrollArea()
        self.operation_scroll.setObjectName("privateFrameOperationScroll")
        self.operation_scroll.setFrameShape(QFrame.NoFrame)
        self.operation_scroll.setWidgetResizable(True)
        self.operation_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.operation_scroll.setWidget(self.operation_panel)
        self.operation_panel.installEventFilter(self)
        self.content.addWidget(self.operation_scroll, 1)

        self.video_input = UploadPreview(
            "Input Video",
            extensions=_VIDEO_EXTENSIONS,
            dialog_filter=_video_dialog_filter(context.config.ui_language),
            prompt="Click to choose or drag a video here",
        )
        self.video_input.setObjectName("privateFrameVideoInput")
        self.video_input.setFixedSize(_VIDEO_PREVIEW_SIZE, _VIDEO_PREVIEW_SIZE)
        self.video_input.placeholder.setStyleSheet("font-size: 13px; padding: 8px;")
        self.video_input.file_label.hide()
        self.video_input.pathChanged.connect(self._video_input_changed)

        input_card, input_layout = self.card()
        input_layout.setSpacing(6)
        video_row = QHBoxLayout()
        video_row.setSpacing(14)
        video_row.addWidget(self.video_input, 0, Qt.AlignmentFlag.AlignTop)

        self.video_metadata = QWidget()
        self.video_metadata.setObjectName("privateFrameVideoMetadata")
        metadata_layout = QVBoxLayout(self.video_metadata)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setSpacing(6)
        metadata_title = QLabel("Video information")
        metadata_title.setObjectName("privateFrameVideoMetadataTitle")
        metadata_title.setStyleSheet("font-weight: 650;")
        metadata_layout.addWidget(metadata_title)
        metadata_form = QGridLayout()
        metadata_form.setContentsMargins(0, 0, 0, 0)
        metadata_form.setHorizontalSpacing(12)
        metadata_form.setVerticalSpacing(4)
        self.video_metadata_values: dict[str, QLabel] = {}
        metadata_fields = (
            ("file", "File name"), ("duration", "Duration"),
            ("resolution", "Resolution"), ("fps", "Frame rate"),
            ("frames", "Frame count"), ("size", "File size"),
            ("video_codec", "Video codec"), ("audio", "Audio"),
        )
        for index, (key, label_text) in enumerate(metadata_fields):
            caption = QLabel(label_text)
            caption.setProperty("role", "muted")
            value_label = QLabel("—")
            value_label.setObjectName("privateFrameVideoMetadata" + "".join(part.capitalize() for part in key.split("_")))
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value_label.setWordWrap(True)
            value_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self.video_metadata_values[key] = value_label
            row, column = index // 2, (index % 2) * 2
            metadata_form.addWidget(caption, row, column)
            metadata_form.addWidget(value_label, row, column + 1)
        metadata_form.setColumnStretch(1, 1)
        metadata_form.setColumnStretch(3, 1)
        metadata_layout.addLayout(metadata_form)
        metadata_layout.addStretch(1)
        video_row.addWidget(self.video_metadata, 1, Qt.AlignmentFlag.AlignTop)
        input_layout.addLayout(video_row)

        self.output_dir = QLineEdit(str(_default_privateframe_output_directory()))
        self.output_dir.setObjectName("privateFrameOutputDirectory")
        self.output_dir.setPlaceholderText("Choose an output directory")
        self.output_dir.textChanged.connect(self._update_output_preview)
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self._browse_output_directory)
        set_button_tooltip(self.browse_output_button)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir, 1)
        output_row.addWidget(self.browse_output_button)
        output_form = QFormLayout()
        # Native macOS forms can keep the entire nested row at its size hint.
        # Let the row grow so its path field receives the remaining width.
        output_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        output_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        output_form.addRow("Output Directory", output_row)
        input_layout.addLayout(output_form)
        self.output_preview = QLabel()
        self.output_preview.setObjectName("privateFrameOutputPreview")
        self.output_preview.setWordWrap(True)
        self.output_preview.setProperty("role", "muted")
        self.operation_layout.addWidget(input_card)

        options_card, options_layout = self.card()
        options_layout.setSpacing(6)
        self.options_grid = QGridLayout()
        self.options_grid.setContentsMargins(0, 0, 0, 0)
        self.options_grid.setHorizontalSpacing(10)
        self.options_grid.setVerticalSpacing(8)
        self.model_summary = QWidget()
        self.model_summary.setObjectName("privateFrameGlobalModelSummary")
        model_summary_layout = QVBoxLayout(self.model_summary)
        model_summary_layout.setContentsMargins(0, 0, 0, 0)
        model_summary_layout.setSpacing(2)
        self.model_label = QLabel()
        self.model_label.setObjectName("privateFrameModelPackage")
        self.model_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.model_root_label = QLabel()
        self.model_root_label.setObjectName("privateFrameModelRoot")
        self.model_root_label.setProperty("role", "muted")
        self.model_root_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.model_root_label.setWordWrap(True)
        model_summary_layout.addWidget(self.model_label)
        self.model_root_label.setParent(self.model_summary)
        self.model_root_label.hide()
        self.analysis_mode = QComboBox()
        self.analysis_mode.setObjectName("privateFrameAnalysisMode")
        self.analysis_mode.setProperty("i18nItems", True)
        for value, label, default_label in (
            (15, "Fast (target 15 analysis FPS)", "Fast (default, target 15 analysis FPS)"),
            (30, "Normal (target 30 analysis FPS)", "Normal (default, target 30 analysis FPS)"),
        ):
            source = default_label if value == self._base_value("scan.max_analysis_fps") else label
            self.analysis_mode.addItem(source, value)
        analysis_fps_tooltip = "Regular face detections per second of input video. Higher values check faces more often; lower values reduce work for faster processing but may miss brief appearances. Extra scans may occur. Output video FPS is unchanged. Current default: {default}."
        self.analysis_mode.setToolTip(analysis_fps_tooltip)
        self.redaction_method = QComboBox()
        self.redaction_method.setObjectName("privateFrameRedactionMethod")
        self.redaction_method.setProperty("i18nItems", True)
        self.redaction_method.addItem("Gaussian blur", "gaussian")
        self.redaction_method.addItem("Mosaic", "mosaic")
        self.output_mode = QComboBox()
        self.output_mode.setObjectName("privateFrameOutputMode")
        self.output_mode.setProperty("i18nItems", True)
        self.output_mode.addItem("Video and analysis results", "json_and_video")
        self.output_mode.addItem("Analysis only (render video later)", "json_only")
        self.output_mode.currentIndexChanged.connect(self._output_mode_changed)
        self.recognition_policy = QComboBox()
        self.recognition_policy.setObjectName("privateFrameRecognitionPolicy")
        self.recognition_policy.setProperty("i18nItems", True)
        self.recognition_policy.addItem("Blur everyone", "all")
        self.recognition_policy.addItem("Blur only people in the photos", "blur_only")
        self.recognition_policy.addItem("Keep people in the photos clear", "exempt")
        self.recognition_policy.currentIndexChanged.connect(
            self._recognition_policy_changed
        )
        provider_name, provider_tooltip = provider_runtime_display(
            str(context.config.provider),
            context.config.ui_language,
        )
        self.provider_label = QLabel(provider_name)
        self.provider_label.setObjectName("privateFrameProvider")
        self.provider_label.setToolTip(provider_tooltip)
        self.more_options_button = QPushButton("More Options…")
        self.more_options_button.setObjectName("privateFrameMoreOptionsButton")
        self.more_options_button.clicked.connect(self._show_more_options)
        set_button_tooltip(self.more_options_button)
        self.analysis_fps_label = QLabel("Target analysis FPS")
        self.analysis_fps_label.setToolTip(analysis_fps_tooltip)
        self.preserve_audio = QCheckBox("Preserve original audio when supported")
        self.preserve_audio.setObjectName("privateFramePreserveAudio")
        self.preserve_audio.setChecked(self._base_value("render.video_output.audio.redacted") != "none")
        self.preserve_audio.setToolTip("Existing AAC audio is copied without re-encoding. Other audio formats are omitted; they are not converted automatically.")
        self.preserve_audio.toggled.connect(self._update_audio_notice)
        self.audio_notice = QLabel()
        self.audio_notice.setObjectName("privateFrameAudioNotice")
        self.audio_notice.setWordWrap(True)
        self.audio_notice.setProperty("role", "muted")
        audio_widget = QWidget()
        audio_layout = QHBoxLayout(audio_widget)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.setSpacing(8)
        audio_layout.addWidget(self.preserve_audio)
        self.audio_notice.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        audio_layout.addWidget(self.audio_notice, 1)
        self._primary_option_rows: list[tuple[QLabel, QWidget]] = [
            (QLabel("Redaction"), self.redaction_method),
            (self.analysis_fps_label, self.analysis_mode),
        ]
        for label, _field in self._primary_option_rows:
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            _field.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        policy_form = QFormLayout()
        policy_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        policy_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        policy_form.addRow("Who should be blurred?", self.recognition_policy)
        options_layout.addLayout(policy_form)
        self.selective_privacy_note = QLabel()
        self.selective_privacy_note.setObjectName("privateFrameRecognitionNotice")
        self.selective_privacy_note.setWordWrap(True)
        self.selective_privacy_note.setProperty("role", "muted")
        options_layout.addWidget(self.selective_privacy_note)
        self.reference_panel = QWidget()
        self.reference_panel.setObjectName("privateFrameReferencePanel")
        reference_layout = QVBoxLayout(self.reference_panel)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(4)
        self.reference_dir = QLineEdit()
        self.reference_dir.setObjectName("privateFrameReferenceDirectory")
        self.reference_dir.setPlaceholderText("Choose a folder of reference photos")
        self.reference_dir.editingFinished.connect(self._refresh_reference_photos)
        self.browse_reference_button = QPushButton("Browse")
        self.browse_reference_button.setObjectName("privateFrameBrowseReferenceButton")
        self.browse_reference_button.clicked.connect(self._browse_reference_directory)
        set_button_tooltip(self.browse_reference_button)
        reference_row = QHBoxLayout()
        reference_row.addWidget(self.reference_dir, 1)
        reference_row.addWidget(self.browse_reference_button)
        reference_form = QFormLayout()
        reference_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        reference_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        reference_form.addRow("Reference photos", reference_row)
        reference_layout.addLayout(reference_form)
        self.reference_dir.setToolTip("Each photo uses only its largest face. Use clear single-person photos when possible. JPG, JPEG, PNG and WebP files directly in this folder are included; subfolders are ignored.")
        self.reference_status = QLabel()
        self.reference_status.setObjectName("privateFrameReferenceStatus")
        self.reference_status.setWordWrap(True)
        reference_layout.addWidget(self.reference_status)
        options_layout.addWidget(self.reference_panel)
        options_layout.addLayout(self.options_grid)
        audio_options_row = QHBoxLayout()
        audio_options_row.addWidget(audio_widget, 1)
        audio_options_row.addWidget(self.more_options_button)
        options_layout.addLayout(audio_options_row)
        self.model_status_label = QLabel()
        self.model_status_label.setObjectName("privateFrameModelStatus")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setProperty("role", "status")
        options_layout.addWidget(self.model_status_label)
        self.model_settings_panel = QWidget()
        model_settings_layout = QVBoxLayout(self.model_settings_panel)
        model_settings_layout.setContentsMargins(0, 0, 0, 0)
        model_settings_layout.setSpacing(4)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Global model"))
        model_row.addWidget(self.model_summary, 1)
        model_settings_layout.addLayout(model_row)
        provider_row = QHBoxLayout()
        provider_row.setContentsMargins(0, 2, 0, 0)
        provider_row.addWidget(QLabel("Provider"))
        provider_row.addWidget(self.provider_label)
        provider_row.addStretch(1)
        self.open_models_button = QPushButton("Open Models")
        self.open_models_button.setObjectName("privateFrameOpenModelsButton")
        self.open_models_button.clicked.connect(self._open_models)
        set_button_tooltip(self.open_models_button)
        provider_row.addWidget(self.open_models_button)
        model_settings_layout.addLayout(provider_row)
        self._layout_primary_options(two_columns=True)
        self.operation_layout.addWidget(options_card)
        self.operation_layout.addStretch(1)

        self._create_more_options_dialog()
        self._initialize_default_controls()
        self.execution_panel = QWidget()
        self.execution_panel.setObjectName("privateFrameExecutionPanel")
        self.execution_layout = QVBoxLayout(self.execution_panel)
        self.execution_layout.setContentsMargins(0, 0, 0, 0)
        self.execution_layout.setSpacing(6)
        self.content.addWidget(self.execution_panel)

        self.start_button = QPushButton("Start Processing")
        self.start_button.setObjectName("privateFrameStartButton")
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("privateFrameCancelButton")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)
        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.setObjectName("privateFrameOpenOutputButton")
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.open_output_button.setEnabled(False)
        for button in (
            self.start_button,
            self.cancel_button,
            self.open_output_button,
        ):
            set_button_tooltip(button)
        self.execution_layout.addWidget(
            self.row(
                self.start_button,
                self.cancel_button,
                self.open_output_button,
            )
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("privateFrameProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(tr("Ready", context.config.ui_language))
        self.stage_label = QLabel("Ready")
        self.stage_label.setObjectName("privateFrameStage")
        self.stage_label.setWordWrap(True)
        self.stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.processing_details_button = QPushButton("Processing details…")
        self.processing_details_button.setObjectName("privateFrameProcessingDetailsButton")
        self.processing_details_button.clicked.connect(self._show_processing_details)
        set_button_tooltip(self.processing_details_button)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_bar, 2)
        progress_row.addWidget(self.stage_label, 1)
        progress_row.addWidget(self.processing_details_button)
        self.execution_layout.addLayout(progress_row)

        self.processing_details_dialog = QDialog(self)
        self.processing_details_dialog.setObjectName("privateFrameProcessingDetailsDialog")
        self.processing_details_dialog.setWindowTitle("PrivateFrame Processing Details")
        self.processing_details_dialog.setModal(False)
        self.processing_details_dialog.resize(720, 440)
        details_layout = QVBoxLayout(self.processing_details_dialog)
        details_layout.addWidget(self.output_preview)
        self.summary = QPlainTextEdit()
        self.summary.setObjectName("privateFrameSummary")
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText(
            tr(
                "Processing details and output paths will appear here.",
                context.config.ui_language,
            )
        )
        details_layout.addWidget(self.summary, 1)
        self.close_processing_details_button = QPushButton("Close")
        self.close_processing_details_button.clicked.connect(self.processing_details_dialog.close)
        set_button_tooltip(self.close_processing_details_button)
        close_details_row = QHBoxLayout()
        close_details_row.addStretch(1)
        close_details_row.addWidget(self.close_processing_details_button)
        details_layout.addLayout(close_details_row)
        self._update_output_option_controls()
        self._render_reference_status(context.config.ui_language)
        self._update_output_preview()
        self._refresh_global_model_status()
        self._refresh_default_help(context.config.ui_language)
        if self.recognition_policy.currentData() != "all":
            self._refresh_reference_photos()

    def _create_more_options_dialog(self) -> None:
        """Keep choices when the non-modal dialog is closed; groups can reset."""
        self.more_options_dialog = QDialog(self)
        self.more_options_dialog.setObjectName("privateFrameMoreOptionsDialog")
        self.more_options_dialog.setWindowTitle("PrivateFrame More Options")
        self.more_options_dialog.setModal(False)
        self.more_options_dialog.setMinimumWidth(560)
        layout = QVBoxLayout(self.more_options_dialog)
        layout.addWidget(self.model_settings_panel)
        self.more_option_groups = {}
        self.reset_option_buttons = {}

        def group(key, title):
            box = QGroupBox(title)
            box.setObjectName("privateFrameOptions" + key.capitalize())
            group_layout = QVBoxLayout(box)
            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            group_layout.addLayout(form)
            reset = QPushButton("Restore defaults")
            reset.clicked.connect(lambda _checked=False, group_key=key: self._reset_option_group(group_key))
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(reset)
            group_layout.addLayout(row)
            layout.addWidget(box)
            self.more_option_groups[key] = box
            self.reset_option_buttons[key] = reset
            return form

        appearance = group("appearance", "Redaction appearance")
        self.between_scan_frames = QComboBox()
        self.between_scan_frames.setObjectName("privateFrameBetweenScanFrames")
        self.between_scan_frames.setProperty("i18nItems", True)
        for label, value in (("Automatic", "auto"), ("Interpolate", "interpolate"), ("Visual tracking", "visual")):
            self.between_scan_frames.addItem(label, value)
        self.between_scan_frames.setToolTip("Automatic follows the default tracking method. Interpolation is faster; visual tracking follows motion between detection frames and takes more work.")
        self.box_scale = QComboBox()
        self.box_scale.setObjectName("privateFrameBoxScale")
        self.box_scale.setProperty("i18nItems", True)
        for label, value in (("Standard (1.00×)", 1.0), ("Extra margin (1.15×)", 1.15), ("Maximum margin (1.30×)", 1.30)):
            self.box_scale.addItem(label, value)
        self.box_scale.setToolTip("A larger margin blurs more area around each detected face.")
        appearance.addRow("Face coverage", self.box_scale)
        appearance.addRow("Follow faces between detections", self.between_scan_frames)

        video = group("video", "Video output")
        self.video_preset = QComboBox()
        self.video_preset.setObjectName("privateFrameVideoPreset")
        self.video_preset.setProperty("i18nItems", True)
        for label, value in (("Very fast encoding", "veryfast"), ("Medium encoding", "medium"), ("Slow encoding", "slow")):
            self.video_preset.addItem(label, value)
        self.video_preset.setToolTip("Encoding speed changes processing time and file size, not face detection or matching. Slower encoding can produce smaller files at the same quality.")
        self.video_crf = QComboBox()
        self.video_crf.setObjectName("privateFrameVideoCrf")
        self.video_crf.setProperty("i18nItems", True)
        for label, value in (("High quality (CRF 18)", 18), ("Balanced (CRF 23)", 23), ("Smaller file (CRF 28)", 28)):
            self.video_crf.addItem(label, value)
        self.video_crf.setToolTip("Video quality affects compression and file size, not face detection or matching. 18 keeps more detail and makes larger files; 23 is balanced; 28 makes smaller files with less detail.")
        video.addRow("Video quality", self.video_crf)
        video.addRow("Encoding speed", self.video_preset)
        video.addRow("Output content", self.output_mode)

        matching = group("matching", "Person matching")
        self.recognition_profile = QComboBox()
        self.recognition_profile.setObjectName("privateFrameRecognitionProfile")
        self.recognition_profile.setProperty("i18nItems", True)
        for label, value in (("Quick check", "fast"), ("Balanced check", "balanced"), ("Extended check", "accurate")):
            self.recognition_profile.addItem(label, value)
        self.recognition_profile.setToolTip("Checks up to 1, 3 or 5 selected frames per track unless overridden in the configuration. More checks take longer and provide more evidence; they do not guarantee a correct match.")
        self.recognition_threshold = QDoubleSpinBox()
        self.recognition_threshold.setObjectName("privateFrameRecognitionThreshold")
        self.recognition_threshold.setRange(0.0, 1.0)
        self.recognition_threshold.setSingleStep(0.01)
        self.recognition_threshold.setDecimals(2)
        self.recognition_threshold.setToolTip("Minimum face similarity for a match (0–1). Increase it to reduce incorrect matches; decrease it if the right person is often missed. Keep the default ({default}) unless you have reviewed the results.")
        matching.addRow("Matching checks", self.recognition_profile)
        matching.addRow("Match threshold", self.recognition_threshold)
        self._group_defaults = {
            "appearance": [(self.box_scale, "render.redaction.box_scale"), (self.between_scan_frames, "tracking.between_scan_frames")],
            "video": [(self.video_crf, "render.video_output.rate_control"), (self.video_preset, "render.video_output.preset"), (self.output_mode, None)],
            "matching": [(self.recognition_profile, "recognition.profile"), (self.recognition_threshold, "recognition.similarity_threshold")],
        }
        for values in self._group_defaults.values():
            for widget, _value in values:
                if isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(self._update_more_options_marker)
                else:
                    widget.valueChanged.connect(self._update_more_options_marker)
        self.close_more_options_button = QPushButton("Close")
        self.close_more_options_button.setObjectName("privateFrameCloseMoreOptionsButton")
        self.close_more_options_button.clicked.connect(self.more_options_dialog.close)
        set_button_tooltip(self.close_more_options_button)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_row.addWidget(self.close_more_options_button)
        layout.addLayout(close_row)

    def _base_value(self, path: str) -> Any:
        value = self._base_defaults
        for key in path.split("."):
            value = value[key]
        return value

    def _control_default(self, path: str | None) -> Any:
        # Output content is a GUI workflow choice, not a processing parameter.
        if path is None:
            return "json_and_video"
        value = self._base_value(path)
        if path == "render.video_output.rate_control":
            return value["quality"] if value["mode"] == "crf" else None
        return value

    def _select_configured_value(self, widget: QComboBox, value: Any, *, source: str = "Configured value ({value})", values: dict[str, str] | None = None) -> None:
        index = next((index for index in range(widget.count()) if widget.itemData(index) == value), -1)
        if index < 0:
            if values is None and value in (None, ""):
                source, values = "Automatic", {}
            values = values if values is not None else {"value": str(value)}
            widget.addItem(tr(source, self.context.config.ui_language).format(**values), value)
            index = widget.count() - 1
            widget.setItemData(index, source, Qt.UserRole + 3000)
            self._configured_combo_entries.append((widget, index, source, values))
        widget.setCurrentIndex(index)

    def _initialize_default_controls(self) -> None:
        fields = [
            (self.analysis_mode, "scan.max_analysis_fps"),
            (self.redaction_method, "render.redaction.method"),
            (self.recognition_policy, "recognition.mode"),
            *(entry for group in self._group_defaults.values() for entry in group),
        ]
        for widget, path in fields:
            previous = widget.blockSignals(True)
            value = self._control_default(path)
            if isinstance(widget, QComboBox):
                if path == "render.video_output.rate_control" and value is None:
                    self._select_configured_value(widget, None, source="From configuration ({mode})", values={"mode": str(self._base_value(path)["mode"])})
                else:
                    self._select_configured_value(widget, value)
            else:
                widget.setDecimals(max(2, -Decimal(str(value)).as_tuple().exponent))
                widget.setValue(value)
            widget.blockSignals(previous)
        reference = self._base_value("recognition.reference_dir")
        if reference is not None:
            reference_path = Path(str(reference)).expanduser()
            if not reference_path.is_absolute():
                reference_path = DEFAULT_CONFIG_PATH.parent / reference_path
            self.reference_dir.setText(str(reference_path))
        else:
            self.reference_dir.clear()

    def _reset_option_group(self, group: str) -> None:
        for widget, path in self._group_defaults[group]:
            value = self._control_default(path)
            if isinstance(widget, QComboBox):
                self._select_configured_value(widget, value)
            else:
                widget.setValue(value)
        self._update_more_options_marker()

    def _refresh_default_help(self, language: str | None) -> None:
        analysis_source = "Regular face detections per second of input video. Higher values check faces more often; lower values reduce work for faster processing but may miss brief appearances. Extra scans may occur. Output video FPS is unchanged. Current default: {default}."
        analysis_text = tr(analysis_source, language).format(default=format(float(self._base_value("scan.max_analysis_fps")), "g"))
        self.analysis_mode.setToolTip(analysis_text)
        self.analysis_fps_label.setToolTip(analysis_text)
        threshold_source = "Minimum face similarity for a match (0–1). Increase it to reduce incorrect matches; decrease it if the right person is often missed. Keep the default ({default}) unless you have reviewed the results."
        self.recognition_threshold.setToolTip(tr(threshold_source, language).format(default=format(float(self._base_value("recognition.similarity_threshold")), "g")))
        for widget, index, source, values in self._configured_combo_entries:
            widget.setItemText(index, tr(source, language).format(**values))

    def _current_control_value(self, widget: QWidget, path: str | None) -> Any:
        value = widget.currentData() if isinstance(widget, QComboBox) else widget.value()
        if path == "tracking.between_scan_frames" and value == "auto":
            return self._control_default(path)
        return value

    def _update_more_options_marker(self, *_args, language: str | None = None) -> None:
        if not hasattr(self, "_group_defaults"):
            return
        changed = False
        for group, fields in self._group_defaults.items():
            if group == "matching" and self.recognition_policy.currentData() == "all":
                continue
            if any(self._current_control_value(widget, path) != self._control_default(path) for widget, path in fields):
                changed = True
                break
        source = "More Options… (modified)" if changed else "More Options…"
        self.more_options_button.setProperty("_insightface_i18n_source", source)
        self.more_options_button.setText(tr(source, language or self.context.config.ui_language))

    def _show_more_options(self) -> None:
        self.more_options_dialog.show()
        self.more_options_dialog.raise_()
        self.more_options_dialog.activateWindow()

    def _show_processing_details(self) -> None:
        self.processing_details_dialog.show()
        self.processing_details_dialog.raise_()
        self.processing_details_dialog.activateWindow()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "options_grid"):
            self._fit_primary_options(event.size().width())

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if (
            watched is getattr(self, "operation_panel", None)
            and event.type() == QEvent.Type.LayoutRequest
            and hasattr(self, "_primary_option_rows")
        ):
            # Native styles can update translated size hints after resizeEvent.
            # Reflow when those hints settle even if the page width is unchanged.
            self._fit_primary_options(self.width())
        return super().eventFilter(watched, event)

    def _fit_primary_options(self, width: int) -> None:
        # Measure translated controls instead of reserving two rows at a fixed
        # window width. Reserve scrollbar space so showing it cannot cause a
        # repeated one-column/two-column layout switch.
        page_margins = self.root_layout.contentsMargins()
        card_margins = self.options_grid.parentWidget().layout().contentsMargins()
        required = sum(
            label.minimumSizeHint().width() + field.minimumSizeHint().width()
            for label, field in self._primary_option_rows
        )
        required += 3 * self.options_grid.horizontalSpacing()
        required += page_margins.left() + page_margins.right()
        required += card_margins.left() + card_margins.right() + 2
        required += self.operation_scroll.verticalScrollBar().sizeHint().width()
        self._layout_primary_options(two_columns=width >= required)

    def _layout_primary_options(self, two_columns: bool) -> None:
        """Lay out primary controls in one or two columns without recreating them."""

        two_columns = bool(two_columns)
        if self._primary_options_two_columns == two_columns:
            return
        self._primary_options_two_columns = two_columns
        for label, field in self._primary_option_rows:
            self.options_grid.removeWidget(label)
            self.options_grid.removeWidget(field)

        for index, (label, field) in enumerate(self._primary_option_rows):
            if two_columns:
                row = index // 2
                label_column = (index % 2) * 2
            else:
                row = index
                label_column = 0
            self.options_grid.addWidget(
                label,
                row,
                label_column,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            self.options_grid.addWidget(field, row, label_column + 1)

        for column in range(4):
            self.options_grid.setColumnStretch(column, 0)
        self.options_grid.setColumnStretch(1, 1)
        if two_columns:
            self.options_grid.setColumnStretch(3, 1)

    def _video_input_changed(self, path: str) -> None:
        """Update output names immediately and load preview data in the background."""

        self._update_output_preview()
        self._video_preview_generation += 1
        generation = self._video_preview_generation
        previous_worker = self._video_preview_worker
        self._video_preview_worker = None
        if previous_worker is not None:
            previous_worker.cancel()

        self.video_input.file_label.hide()
        if not path:
            self._video_preview_state = "empty"
            self._video_preview_path = None
            self._video_preview_data = None
            self._reset_video_metadata()
            self._render_video_preview_state(self.context.config.ui_language)
            self._update_audio_notice()
            return

        source = Path(path).expanduser().resolve()
        self._video_preview_state = "loading"
        self._video_preview_path = source
        self._video_preview_data = None
        self._set_video_metadata_pending(source)
        self.video_input.set_image(None)
        self._render_video_preview_state(self.context.config.ui_language)
        self._update_audio_notice()

        def task(progress=None, is_cancelled=None):
            if is_cancelled is not None and is_cancelled():
                raise InterruptedError("Video preview loading was cancelled")
            result = _read_video_preview(source)
            if is_cancelled is not None and is_cancelled():
                raise InterruptedError("Video preview loading was cancelled")
            return result

        def on_result(result):
            self._apply_video_preview(generation, result)

        def on_error(message):
            self._video_preview_failed(generation, source, message)

        def on_finished():
            self._video_preview_finished(generation)
        main = self.window()
        if main is self or not hasattr(main, "run_task"):
            try:
                on_result(task())
            except Exception as exc:
                on_error(str(exc))
            finally:
                on_finished()
            return

        worker = self.run_task(
            "Reading video preview",
            task,
            on_result,
            show_dialog=False,
            on_error=on_error,
            on_finished=on_finished,
        )
        if generation == self._video_preview_generation:
            self._video_preview_worker = worker

    def _reset_video_metadata(self) -> None:
        for label in self.video_metadata_values.values():
            label.setText("—")
            label.setToolTip("")

    def _set_video_metadata_pending(self, source: Path) -> None:
        self._reset_video_metadata()
        language = self.context.config.ui_language
        self.video_metadata_values["file"].setText(source.name)
        self.video_metadata_values["file"].setToolTip(str(source))
        self.video_metadata_values["resolution"].setText(tr("Reading…", language))

    def _apply_video_preview(
        self,
        generation: int,
        preview: VideoPreviewData,
    ) -> None:
        if generation != self._video_preview_generation:
            return
        selected = self.video_input.path()
        if not selected or Path(selected).expanduser().resolve() != preview.path:
            return

        language = self.context.config.ui_language
        self._video_preview_state = "ready"
        self._video_preview_path = preview.path
        self._video_preview_data = preview
        self.video_input.set_image(preview.image, str(preview.path))
        self.video_input.file_label.hide()
        self._render_video_preview_metadata(preview, language)
        self._update_audio_notice(language=language)

        main = self.window()
        if main is not self and hasattr(main, "set_status"):
            main.set_status("Video preview ready.")

    def _render_video_preview_metadata(
        self,
        preview: VideoPreviewData,
        language: str | None,
    ) -> None:
        values = self.video_metadata_values
        values["file"].setText(preview.path.name)
        values["file"].setToolTip(str(preview.path))
        values["resolution"].setText(f"{preview.width} × {preview.height}")
        fps_text = f"{preview.fps:.3f}".rstrip("0").rstrip(".")
        values["fps"].setText(f"{fps_text} FPS")
        values["frames"].setText(f"{preview.frame_count:,}")
        values["duration"].setText(_format_video_duration(preview.duration))
        values["video_codec"].setText(preview.video_codec or tr("Unknown", language))
        if preview.has_audio is False:
            audio_text = tr("No audio track", language)
        elif preview.audio_codec:
            audio_text = preview.audio_codec
        else:
            audio_text = tr("Unknown", language)
        values["audio"].setText(audio_text)
        values["size"].setText(_format_file_size(preview.file_size))

    def _video_preview_failed(
        self,
        generation: int,
        source: Path,
        _message: str,
    ) -> None:
        if generation != self._video_preview_generation:
            return
        selected = self.video_input.path()
        if not selected or Path(selected).expanduser().resolve() != source:
            return
        self._video_preview_state = "failed"
        self._video_preview_path = source
        self._video_preview_data = None
        self.video_input.set_image(None)
        self._reset_video_metadata()
        self.video_metadata_values["file"].setText(source.name)
        self.video_metadata_values["file"].setToolTip(str(source))
        self._render_video_preview_state(self.context.config.ui_language)
        main = self.window()
        if main is not self and hasattr(main, "set_status"):
            main.set_status("Video preview unavailable.")

    def _video_preview_finished(self, generation: int) -> None:
        if generation == self._video_preview_generation:
            self._video_preview_worker = None

    def _render_video_preview_state(self, language: str | None) -> None:
        self.video_input.file_label.hide()
        state = self._video_preview_state
        source = self._video_preview_path
        if state == "empty":
            self.video_input.placeholder.setText(
                f"{tr(self.video_input.title, language)}\n"
                f"{tr(self.video_input.prompt, language)}"
            )
            self.video_input.stack.setCurrentWidget(self.video_input.placeholder)
        elif state == "loading" and source is not None:
            self.video_input.placeholder.setText(tr("Reading video preview…", language))
            self.video_input.stack.setCurrentWidget(self.video_input.placeholder)
            self.video_metadata_values["resolution"].setText(
                tr("Reading…", language)
            )
        elif state == "failed" and source is not None:
            self.video_input.placeholder.setText(
                "\n".join(
                    (
                        tr("Preview unavailable", language),
                        source.name,
                        tr("Drop another file to replace it", language),
                    )
                )
            )
            self.video_input.stack.setCurrentWidget(self.video_input.placeholder)
            self.video_input.file_label.hide()
            self.video_metadata_values["resolution"].setText(
                tr("Unavailable", language)
            )
        elif state == "ready" and self._video_preview_data is not None:
            self._render_video_preview_metadata(self._video_preview_data, language)
            self.video_input.stack.setCurrentWidget(self.video_input.viewer)

    def _browse_reference_directory(self) -> None:
        initial = self.reference_dir.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, tr("Select Reference Photo Folder", self.context.config.ui_language), str(Path(initial).expanduser()))
        if folder:
            self.reference_dir.setText(folder)
            self._refresh_reference_photos()

    def _refresh_reference_photos(self, language: str | None = None) -> None:
        if self.recognition_policy.currentData() == "all":
            return
        try:
            _root, images = _scan_reference_folder(self.reference_dir.text())
        except (OSError, ValueError) as error:
            self._set_reference_status(str(error), language=language)
        else:
            self._set_reference_status("Found {count} reference photos. Faces will be checked when processing starts.", language=language, count=len(images))

    def _set_reference_status(self, source: str, *, language: str | None = None, **values: object) -> None:
        self._reference_status_source = source
        self._reference_status_values = dict(values)
        self._render_reference_status(language or self.context.config.ui_language)

    def _render_reference_status(self, language: str | None) -> None:
        text = tr(self._reference_status_source, language)
        if self._reference_status_values:
            text = text.format(**self._reference_status_values)
        self.reference_status.setText(text)

    @Slot(object)
    def _reference_log_received(self, event: dict[str, Any]) -> None:
        self._reference_log_entries.append(dict(event))
        if event.get("kind") == "summary":
            self._set_reference_status("Reference photos: {used} used, {skipped} skipped ({total} total).", **{key: event[key] for key in ("used", "skipped", "total")})
        self.summary.appendPlainText(_reference_log_text(event, self.context.config.ui_language))

    def _browse_output_directory(self) -> None:
        initial = self.output_dir.text().strip() or str(
            _default_privateframe_output_directory()
        )
        folder = QFileDialog.getExistingDirectory(
            self,
            tr("Select Output Directory", self.context.config.ui_language),
            str(Path(initial).expanduser()),
        )
        if folder:
            self.output_dir.setText(folder)

    def _global_model_status(self) -> PrivateFrameModelStatus:
        return inspect_privateframe_model(
            str(self.context.config.model_name),
            str(self.context.config.model_root),
            require_recognition=self.recognition_policy.currentData() != "all",
        )

    def _refresh_global_model_status(
        self,
        language: str | None = None,
    ) -> PrivateFrameModelStatus:
        language = language or self.context.config.ui_language
        if self._running and self._last_job is not None:
            job = self._last_job
            self.model_label.setText(job.model_name)
            self.model_root_label.setText(str(job.model_root))
            self.model_label.setToolTip(str(job.model_root))
            self.model_root_label.setToolTip(str(job.model_root))
            current_name = str(self.context.config.model_name)
            current_root = Path(self.context.config.model_root).expanduser().resolve()
            if current_name != job.model_name or current_root != job.model_root:
                self.model_status_label.show()
                self.model_status_label.setText(
                    tr(
                        "This run is using its startup model snapshot. The new global "
                        "model selection applies to the next run.",
                        language,
                    )
                )
            else:
                self.model_status_label.hide()
                self.model_status_label.setText(
                    tr(
                        "This run is using the global model snapshot captured at startup.",
                        language,
                    )
                )
            # The return value is used only for button state. A running job has
            # already passed compatibility validation.
            return PrivateFrameModelStatus(
                model_name=job.model_name,
                model_root=job.model_root,
                package_path=job.model_root / "models" / job.model_name,
                state="running",
                can_start=False,
                message=self.model_status_label.text(),
            )

        status = self._global_model_status()
        self._update_model_requirement_state(status, language)
        self.model_label.setText(status.model_name or "—")
        self.model_root_label.setText(str(status.model_root))
        self.model_label.setToolTip(str(status.model_root))
        self.model_root_label.setToolTip(str(status.model_root))
        download_count = context_activity_count(
            self.context, "model_downloads_in_progress"
        )
        if download_count:
            self.model_status_label.show()
            self.model_status_label.setText(
                tr(
                    "A model download is in progress. Wait for it to finish before "
                    "starting PrivateFrame.",
                    language,
                )
            )
            self.start_button.setEnabled(False)
        else:
            self.model_status_label.setVisible(status.state not in {"ready", "unsupported"})
            self.model_status_label.setText(tr("Ready", language) if status.state == "ready" else _localized_model_status(status, language))
            self.model_status_label.setToolTip(_localized_model_status(status, language))
            self.start_button.setEnabled(status.can_start)
        return status

    def _update_model_requirement_state(
        self,
        status: PrivateFrameModelStatus,
        language: str | None = None,
    ) -> None:
        """Block the workspace when the global model cannot run PrivateFrame."""

        unsupported = status.state == "unsupported"
        self.model_requirement_banner.setVisible(unsupported)
        self.operation_panel.setEnabled(not unsupported)
        self.operation_opacity.setOpacity(
            _UNAVAILABLE_CONTENT_OPACITY if unsupported else 1.0
        )
        self.more_options_dialog.setEnabled(not unsupported)
        if not unsupported:
            return
        if self.more_options_dialog.isVisible():
            self.more_options_dialog.close()
        language = language or self.context.config.ui_language
        self.model_requirement_message.setProperty(
            "_insightface_i18n_source", status.message
        )
        self.model_requirement_message.setText(
            tr(status.message, language)
        )
        self.model_requirement_current_model_value.setText(
            status.model_name or "—"
        )

    def _open_models(self) -> None:
        host = self.window()
        if host is not self and hasattr(host, "open_model_manager"):
            host.open_model_manager("Model Settings")
            return
        self.set_status("Open Models from the main application window.")

    def _selected_job(self) -> PrivateFrameJob:
        if context_activity_count(
            self.context, "model_downloads_in_progress"
        ):
            raise RuntimeError(
                "Wait for the model download to finish before starting PrivateFrame."
            )
        source = self.video_input.path()
        if not source:
            raise ValueError("Select an input video first.")
        if not Path(source).expanduser().is_file():
            raise ValueError("The selected input video does not exist.")
        output = self.output_dir.text().strip()
        if not output:
            raise ValueError("Select an output directory.")
        model_status = self._global_model_status()
        if not model_status.can_start:
            raise ValueError(
                _localized_model_status(
                    model_status,
                    self.context.config.ui_language,
                )
            )
        recognition_mode = str(self.recognition_policy.currentData())
        job = build_privateframe_job(
            input_path=source,
            output_dir=output,
            model_package=str(self.context.config.model_name),
            model_root=str(self.context.config.model_root),
            max_analysis_fps=float(self.analysis_mode.currentData()),
            redaction_method=str(self.redaction_method.currentData()),
            runtime_provider=str(self.context.config.provider),
            preserve_aac_audio=(None if self.preserve_audio.isChecked() == (self._base_value("render.video_output.audio.redacted") != "none") else self.preserve_audio.isChecked()),
            output_mode=str(self.output_mode.currentData()),
            between_scan_frames=str(self.between_scan_frames.currentData()),
            box_scale=float(self.box_scale.currentData()),
            video_preset=self.video_preset.currentData(),
            video_crf=self.video_crf.currentData(),
            recognition_mode=recognition_mode,
            recognition_reference_dir=(
                self.reference_dir.text().strip() if recognition_mode != "all" else None
            ),
            recognition_profile=str(self.recognition_profile.currentData()),
            recognition_similarity_threshold=self.recognition_threshold.value(),
            base_defaults=self._base_defaults,
        )
        if not job.config_path.is_file():
            raise RuntimeError(
                tr(
                    "PrivateFrame configuration is missing: {path}",
                    self.context.config.ui_language,
                ).format(path=job.config_path)
            )
        return job

    def _update_output_preview(
        self,
        *_args,
        language: str | None = None,
    ) -> None:
        language = language or self.context.config.ui_language
        source = self.video_input.path()
        output = self.output_dir.text().strip()
        if not source or not output:
            self.output_preview.setText(
                tr(
                    "Select a video and output directory.",
                    language,
                )
            )
            self.output_preview.setToolTip("")
            self.output_dir.setToolTip(self.output_preview.text())
            return
        paths = default_output_paths(source, output)
        lines = [f"{tr('Analysis JSON', language)}: {paths.result_json.name}"]
        if self.output_mode.currentData() == "json_and_video":
            lines.append(f"{tr('Redacted video', language)}: {paths.result_video.name}")
        self.output_preview.setText("\n".join(lines))
        self.output_preview.setToolTip(str(paths.result_json) + ("\n" + str(paths.result_video) if self.output_mode.currentData() == "json_and_video" else ""))
        self.output_dir.setToolTip(self.output_preview.toolTip())

    def _set_summary_text(self, text: str) -> None:
        """Replace the summary and keep the newest lines in view."""

        self.summary.setPlainText(text)
        cursor = self.summary.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.summary.setTextCursor(cursor)
        self.summary.ensureCursorVisible()
        scrollbar = self.summary.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_stage(self, source: str, **values: object) -> None:
        self._stage_source = source
        self._stage_values = dict(values)
        self._render_stage(self.context.config.ui_language)

    def _render_stage(self, language: str | None) -> None:
        text = tr(self._stage_source, language)
        if self._stage_values:
            text = text.format(**self._stage_values)
        self.stage_label.setText(text)

    def _set_progress_format(self, source: str | None) -> None:
        self._progress_format_source = source
        self._render_progress_format(self.context.config.ui_language)

    def _render_progress_format(self, language: str | None) -> None:
        if self._progress_format_source is None:
            self.progress_bar.setFormat("%p%")
        else:
            self.progress_bar.setFormat(tr(self._progress_format_source, language))

    def _render_processing_summary(
        self,
        result: dict[str, Any],
        language: str | None,
    ) -> None:
        analysis = result.get("analysis", {})
        render = result.get("render") or {}
        gui_result = result.get("gui", {})
        source_audio = gui_result.get("source_audio_codec")
        audio_mode = gui_result.get("audio_output_mode")
        render_video = bool(
            self._last_job and self._last_job.output_mode == "json_and_video"
        )
        total_seconds = _pipeline_total_seconds(
            analysis,
            render,
            render_video=render_video,
        )
        lines = [
            tr("PrivateFrame processing completed.", language),
            f"{tr('Analysis JSON', language)}: "
            f"{self._last_job.result_path if self._last_job else ''}",
            tr("Frames: {count}", language).format(
                count=analysis.get("frame_count", render.get("frame_count", ""))
            ),
            tr("Accepted tracks: {count}", language).format(
                count=analysis.get("accepted_tracks", "")
            ),
            tr("Analysis seconds: {seconds}", language).format(
                seconds=analysis.get("timings", {}).get("analysis_seconds", "")
            ),
        ]
        if render_video:
            lines.insert(
                2,
                f"{tr('Redacted video', language)}: {self._last_job.redacted_path}",
            )
            lines.append(
                tr("Render seconds: {seconds}", language).format(
                    seconds=render.get("seconds", "")
                )
            )
            if total_seconds is not None:
                lines.append(
                    tr("Total seconds: {seconds}", language).format(
                        seconds=f"{total_seconds:.2f}"
                    )
                )
            if source_audio is None:
                lines.append(tr("Audio: no source audio track", language))
            elif audio_mode == "aac":
                lines.append(tr("Audio: preserved (AAC)", language))
            else:
                lines.append(
                    tr("Audio: omitted (source codec: {codec})", language).format(
                        codec=source_audio
                    )
                )
        else:
            lines.append(tr("Video rendering: skipped (JSON only)", language))
            if total_seconds is not None:
                lines.append(
                    tr("Total seconds: {seconds}", language).format(
                        seconds=f"{total_seconds:.2f}"
                    )
                )
        if not render_video:
            preference = gui_result.get("audio_preference", "aac")
            lines.append(tr("Audio preference for later rendering: preserve AAC when available." if preference == "aac" else "Audio preference for later rendering: no audio.", language))
        events = gui_result.get("reference_logs", self._reference_log_entries)
        lines.extend(_reference_log_text(event, language) for event in events)
        recognition = analysis.get("recognition", {})
        if recognition.get("enabled") and not any(record.get("status") == "CONFIRMED" for record in recognition.get("tracks", {}).values()):
            lines.append(tr("Video did not match any reference people.", language))
        self._set_summary_text("\n".join(lines))

    def retranslate_dynamic_content(self, language: str | None) -> None:
        """Rebuild stateful PrivateFrame text after a live language change."""

        self.summary.setPlaceholderText(
            tr(
                "Processing details and output paths will appear here.",
                language,
            )
        )
        self.video_input.dialog_filter = _video_dialog_filter(language)
        self._render_video_preview_state(language)
        self._update_output_preview(language=language)
        self._refresh_global_model_status(language)
        self._refresh_provider_display(language)
        self._render_reference_status(language)
        self._update_policy_note(language)
        self._update_audio_notice(language=language)
        self._update_more_options_marker(language=language)
        self._refresh_default_help(language)
        self._fit_primary_options(self.width())
        self._render_stage(language)
        self._render_progress_format(language)
        if self._summary_state == "completed" and self._summary_result is not None:
            self._render_processing_summary(self._summary_result, language)
        elif self._summary_state == "cancelled":
            lines = [_reference_log_text(event, language) for event in self._reference_log_entries]
            lines.append(
                tr(
                    "Processing cancelled. Partial work files may remain in the work directory.",
                    language,
                )
            )
            self._set_summary_text("\n".join(lines))
        elif self._summary_state == "error":
            self._set_summary_text("\n".join([*(_reference_log_text(event, language) for event in self._reference_log_entries), (tr(self._summary_error, language).format(**self._summary_error_values) if self._summary_error_values else tr(self._summary_error, language))]))
        elif self._reference_log_entries:
            self._set_summary_text("\n".join(_reference_log_text(event, language) for event in self._reference_log_entries))

    def _output_mode_changed(self, *_args) -> None:
        self._update_output_option_controls()
        self._update_output_preview()

    def _recognition_policy_changed(self, *_args) -> None:
        self._update_output_option_controls()
        self._refresh_global_model_status()
        if self.recognition_policy.currentData() != "all" and self.reference_dir.text().strip():
            self._refresh_reference_photos()

    def _update_policy_note(self, language: str | None = None) -> None:
        source = {
            "all": "All detected faces are blurred. Reference photos are not needed.",
            "blur_only": "Only people matched to your photos are blurred. Unmatched faces stay clear; a missed match may leave a selected person visible.",
            "exempt": "People confidently matched to your photos stay clear. Everyone else, including uncertain matches, is blurred.",
        }[str(self.recognition_policy.currentData())]
        self.selective_privacy_note.setProperty("_insightface_i18n_source", source)
        language = language or self.context.config.ui_language
        text = tr(source, language)
        mode = str(self.recognition_policy.currentData())
        unknown_action = self._base_value("recognition.unknown_action")
        if mode != "all" and unknown_action != "auto":
            fallback = resolve_unknown_action(mode, unknown_action)
            # The main mode still describes confirmed matches; surface any
            # configured exception to its usual unconfirmed-face behavior.
            explanation = "Unconfirmed faces will be blurred, as set in the configuration." if fallback == "blur" else "Unconfirmed faces will stay clear, as set in the configuration."
            if fallback != resolve_unknown_action(mode, "auto"):
                text = tr(explanation, language)
        self.selective_privacy_note.setText(text)
        self.selective_privacy_note.setVisible(mode != "all")
        self.recognition_policy.setToolTip(text)

    def _update_audio_notice(self, *_args, language: str | None = None) -> None:
        if not hasattr(self, "audio_notice"):
            return
        preview = self._video_preview_data
        values = {}
        if self.output_mode.currentData() == "json_only":
            source = "Audio preference is saved for later video rendering."
        elif not self.preserve_audio.isChecked():
            source = "The output video will be silent."
        elif preview is not None and preview.has_audio is False:
            source = "This video has no audio track."
        elif preview is not None and preview.audio_codec not in {None, "aac"}:
            source = "Source audio is {codec}; this format cannot be preserved and the output video will be silent."
            values["codec"] = preview.audio_codec
        elif preview is not None and preview.audio_codec == "aac":
            source = "AAC audio will be preserved."
        else:
            source = ""
        self.audio_notice.setVisible(bool(source))
        self.audio_notice.setText(tr(source, language or self.context.config.ui_language).format(**values))

    def _update_output_option_controls(self) -> None:
        if not hasattr(self, "reference_panel") or not hasattr(self, "_group_defaults"):
            return
        enabled = not self._running
        self.preserve_audio.setEnabled(enabled)
        selective = self.recognition_policy.currentData() != "all"
        self.reference_panel.setVisible(selective)
        self.reference_panel.setEnabled(enabled)
        for group, box in self.more_option_groups.items():
            box.setEnabled(enabled)
            box.setVisible(group != "matching" or selective)
        self._update_policy_note()
        self._update_audio_notice()
        self._update_more_options_marker()
        if self.more_options_dialog.isVisible():
            self.more_options_dialog.adjustSize()

    def start_processing(self) -> None:
        if self._running:
            return
        if context_activity_count(
            self.context, "model_downloads_in_progress"
        ):
            self.show_error(
                "Wait for the model download to finish before starting PrivateFrame."
            )
            return
        try:
            job = self._selected_job()
            job.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.show_error(str(exc))
            return

        existing = [
            path
            for path in (job.result_path, job.redacted_path)
            if path is not None and path.exists()
        ]
        if existing:
            answer = QMessageBox.question(
                self,
                tr("Replace Existing Output", self.context.config.ui_language),
                tr(
                    "The selected output already exists and will be replaced. Continue?",
                    self.context.config.ui_language,
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self._last_job = job
        self._cancel_requested = False
        begin_context_activity(self.context, "privateframe_jobs_in_progress")
        self._privateframe_activity_registered = True
        self._set_running(True)
        self._refresh_global_model_status()
        self.progress_bar.setRange(0, 0)
        self._set_progress_format("Preparing models…")
        if job.output_mode == "json_only":
            self._set_stage("Preparing models and video analysis…")
        else:
            self._set_stage("Preparing models, analysis, and rendering…")
        self._summary_state = "empty"
        self._summary_result = None
        self._summary_error = ""
        self._resolved_provider = ""
        self.summary.clear()
        self._reference_log_entries = []

        def task(progress=None, is_cancelled=None):
            return run_privateframe_job(
                job,
                progress=progress,
                is_cancelled=is_cancelled,
                reference_log=self.reference_log_received.emit,
            )

        try:
            worker = self.run_task(
                "PrivateFrame video processing",
                task,
                self._processing_complete,
                show_dialog=False,
                on_progress=self._processing_progress,
                on_error=self._processing_error,
                on_finished=self._processing_finished,
            )
        except Exception as exc:
            self._processing_finished()
            self.show_error(str(exc))
            return
        if self._running:
            self._worker = worker

    def cancel_processing(self) -> None:
        if not self._running:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self._set_stage("Cancelling after the current frame…")
        if self._worker is not None:
            self._worker.cancel()

    def _processing_progress(self, current: int, total: int, message: str) -> None:
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self._set_progress_format(None)
        if message == "analysis":
            json_only = (
                self._last_job is not None and self._last_job.output_mode == "json_only"
            )
            analysis_total = total if json_only else max(1, total // 2)
            self._set_stage(
                "Analyzing video frames: {current}/{total}",
                current=min(current, analysis_total),
                total=analysis_total,
            )
        elif message == "render":
            analysis_total = total // 2
            render_total = max(1, total - analysis_total)
            self._set_stage(
                "Rendering output video: {current}/{total}",
                current=max(0, current - analysis_total),
                total=render_total,
            )
        elif message:
            self._set_stage(message)

    def _processing_complete(self, result: dict[str, Any]) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._set_progress_format("Completed")
        self._set_stage("PrivateFrame processing completed.")
        self.open_output_button.setEnabled(True)
        analysis = result.get("analysis", {})
        resolved_provider = str(analysis.get("provider", "")).strip()
        self._resolved_provider = resolved_provider
        self._summary_state = "completed"
        self._summary_result = result
        self._summary_error = ""
        self._refresh_provider_display(self.context.config.ui_language)
        self._render_processing_summary(result, self.context.config.ui_language)
        self.set_status("PrivateFrame processing completed.")

    def _processing_error(self, message: str) -> None:
        message, error_values = _privateframe_error_parts(message)
        if self._cancel_requested or "cancel" in message.lower():
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self._set_progress_format("Cancelled")
            self._set_stage("PrivateFrame processing was cancelled.")
            self._summary_state = "cancelled"
            self._summary_result = None
            self._summary_error = ""
            lines = [_reference_log_text(event, self.context.config.ui_language) for event in self._reference_log_entries]
            lines.append(
                tr(
                    "Processing cancelled. Partial work files may remain in the work directory.",
                    self.context.config.ui_language,
                )
            )
            self._set_summary_text("\n".join(lines))
            self.set_status("PrivateFrame processing was cancelled.")
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_progress_format("Failed")
        self._set_stage("PrivateFrame processing failed.")
        self._summary_state = "error"
        self._summary_result = None
        self._summary_error = message
        self._summary_error_values = error_values
        self._set_summary_text("\n".join([*(_reference_log_text(event, self.context.config.ui_language) for event in self._reference_log_entries), (tr(message, self.context.config.ui_language).format(**error_values) if error_values else tr(message, self.context.config.ui_language))]))
        self.show_error(tr(message, self.context.config.ui_language).format(**error_values) if error_values else message)

    def _processing_finished(self) -> None:
        self._worker = None
        if self._privateframe_activity_registered:
            end_context_activity(self.context, "privateframe_jobs_in_progress")
            self._privateframe_activity_registered = False
        self._set_running(False)
        self._cancel_requested = False
        self.refresh()

    def _set_running(self, running: bool) -> None:
        self._running = running
        for widget in (
            self.video_input,
            self.output_dir,
            self.browse_output_button,
            self.analysis_mode,
            self.recognition_policy,
            self.redaction_method,
            self.output_mode,
            self.more_options_button,
            self.preserve_audio,
            self.open_models_button,
        ):
            widget.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self._update_output_option_controls()
        if running:
            self.open_output_button.setEnabled(False)
        else:
            self._refresh_global_model_status()

    def open_output_folder(self) -> None:
        directory = (
            self._last_job.output_dir
            if self._last_job
            else Path(self.output_dir.text()).expanduser()
        )
        if not directory.is_dir():
            self.show_error("The output directory does not exist.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def refresh(self) -> None:
        self._refresh_global_model_status()
        self._refresh_provider_display()
        self._update_output_preview()

    def _refresh_provider_display(self, language: str | None = None) -> None:
        language = language or self.context.config.ui_language
        show_completed_provider = bool(
            self._summary_state == "completed" and self._resolved_provider
        )
        provider_choice = (
            self._last_job.provider_choice
            if self._last_job is not None
            and (self._running or show_completed_provider)
            else str(self.context.config.provider)
        )
        provider_name, provider_tooltip = provider_runtime_display(
            provider_choice,
            language,
        )
        if show_completed_provider:
            provider_name = self._resolved_provider
            provider_tooltip = (
                tr(
                    "Last PrivateFrame run confirmed {provider}.",
                    language,
                ).format(provider=provider_name)
                + " "
                + provider_tooltip
            )
        self.provider_label.setText(provider_name)
        self.provider_label.setToolTip(provider_tooltip)


__all__ = [
    "PrivateFrameJob",
    "PrivateFramePage",
    "build_privateframe_job",
    "run_privateframe_job",
]
