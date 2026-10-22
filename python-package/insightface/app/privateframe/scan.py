"""Multi-view SCRFD detection for frames selected by the streaming pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import cv2
import numpy as np

from .geometry import clip, inverse_cardinal_points, nms_records
from .model_catalog import DETECTION_TASK
from .models import (
    active_face_detector,
    detect_faces,
    detector_max_detections,
    make_face_analysis,
    rotate_image,
)


def _inverse_box(value: np.ndarray, angle: int, original_width: int, original_height: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(item) for item in value)
    if angle == 0:
        return np.asarray([x1, y1, x2, y2], dtype=np.float64)
    if angle == 90:
        return np.asarray([original_width - y2, x1, original_width - y1, x2], dtype=np.float64)
    if angle == -90:
        return np.asarray([y1, original_height - x2, y2, original_height - x1], dtype=np.float64)
    raise ValueError(f"unsupported scan angle {angle}")


def _padded(frame: np.ndarray, horizontal: float, vertical: float) -> tuple[np.ndarray, int, int]:
    px = round(frame.shape[1] * horizontal)
    py = round(frame.shape[0] * vertical)
    return cv2.copyMakeBorder(frame, py, py, px, px, cv2.BORDER_CONSTANT, value=0), px, py


def _candidate_geometry(
    box: np.ndarray,
    source_shape: tuple[int, ...],
) -> tuple[float, float]:
    """Return unclipped area/frame-area and upright height/width."""

    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0, 0.0
    frame_area = max(1.0, float(source_shape[0] * source_shape[1]))
    return width * height / frame_area, height / width


def _candidate_allowed(
    box: np.ndarray,
    source_shape: tuple[int, ...],
    settings: dict[str, Any] | None,
) -> tuple[bool, float, float]:
    """Apply an optional pass-local gate before clipping, NMS and review."""

    area_fraction, height_width_ratio = _candidate_geometry(box, source_shape)
    if not settings or not bool(settings.get("enabled", False)):
        return True, area_fraction, height_width_ratio
    maximum_ratio = settings.get("max_height_width_ratio")
    aspect_allowed = height_width_ratio >= float(settings["min_height_width_ratio"]) and (
        maximum_ratio is None or height_width_ratio <= float(maximum_ratio)
    )
    aspect_exempt = area_fraction >= float(settings.get("aspect_ratio_exempt_min_area_fraction", 1.0))
    allowed = area_fraction >= float(settings["min_box_area_fraction"]) and (aspect_allowed or aspect_exempt)
    return allowed, area_fraction, height_width_ratio


class _Future(Protocol):
    def result(self) -> Any: ...


class ScanRunner:
    def __init__(
        self,
        config: dict[str, Any],
        detector: Any | None = None,
        *,
        face_analysis: Any | None = None,
    ):
        self.config = config
        self.detector_id, _model = active_face_detector(config)
        self.max_detections = detector_max_detections(config)
        self.scan_passes = list(config["scan"]["passes"])
        self.views: list[tuple[dict[str, Any], int]] = []
        sharing = str(config["scan"].get("session_sharing", "single_session_parallel"))
        if sharing not in {"single_session_parallel", "single_session_serial"}:
            raise ValueError(f"unsupported scan.session_sharing: {sharing}")
        # By default SCRFD keeps one fixed-input-shape Session per resolution
        # on every provider, shared by all angles at that resolution. Runtime
        # configuration can instead route every size through the main Session.
        self.face_analysis = face_analysis or make_face_analysis(config)
        self.detector = self.face_analysis.models[DETECTION_TASK]
        if detector is not None and self.detector is not detector:
            raise ValueError("injected detector does not match FaceAnalysis detection task model")
        for scan_pass in self.scan_passes:
            for angle in scan_pass["angles"]:
                self.views.append((scan_pass, int(angle)))
        self.workers = (
            1
            if sharing == "single_session_serial"
            # With pipeline_depth > 1, workers beyond the four views of one
            # frame can preprocess and execute the following frame in flight.
            else max(1, int(config["scan"].get("workers", len(self.views))))
        )
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="pf-scan")

    def _run_view(
        self,
        prepared: _Future,
        entry: tuple[dict[str, Any], int],
        source_shape: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        scan_pass, angle = entry
        canvas, pad_x, pad_y = prepared.result()
        rotated = rotate_image(canvas, angle)
        detections = detect_faces(
            self.face_analysis,
            rotated,
            input_sizes=[int(scan_pass["input_size"])],
            confidence_threshold=float(scan_pass["confidence_threshold"]),
            max_detections=self.max_detections,
        )
        result: list[dict[str, Any]] = []
        for raw in detections:
            canvas_box = _inverse_box(np.asarray(raw["box"]), angle, canvas.shape[1], canvas.shape[0])
            source_unclipped = canvas_box - np.asarray([pad_x, pad_y, pad_x, pad_y], dtype=np.float64)
            allowed, area_fraction, height_width_ratio = _candidate_allowed(
                source_unclipped,
                source_shape,
                scan_pass.get("candidate_filter"),
            )
            if not allowed:
                continue
            source = clip(source_unclipped, source_shape[1], source_shape[0])
            if source[2] - source[0] < 1.0 or source[3] - source[1] < 1.0:
                continue
            record = {
                "box": source.tolist(),
                # Keep the detector's complete source-coordinate geometry
                # paired with its landmarks. ``box`` is clipped for tracking
                # and may be refined later, while this value is immutable and
                # may legitimately extend beyond the source-frame boundary.
                "detector_box": source_unclipped.tolist(),
                "confidence": float(raw["confidence"]),
                "detector": self.detector_id,
                "scan_pass": str(scan_pass["name"]),
                "scan_angle_degrees": angle,
                "detector_size": int(scan_pass["input_size"]),
                "candidate_area_fraction": area_fraction,
                "candidate_height_width_ratio": height_width_ratio,
            }
            if raw.get("landmarks") is not None:
                landmarks = inverse_cardinal_points(raw["landmarks"], angle, canvas.shape[1], canvas.shape[0])
                landmarks -= np.asarray([pad_x, pad_y], dtype=np.float64)
                record["detector_landmarks"] = landmarks.tolist()
            result.append(record)
        return result

    def submit(self, frame: np.ndarray) -> list[_Future]:
        prepared = {
            id(scan_pass): self.executor.submit(
                _padded,
                frame,
                float(scan_pass["horizontal_padding_ratio"]),
                float(scan_pass["vertical_padding_ratio"]),
            )
            for scan_pass in self.scan_passes
        }
        return [
            self.executor.submit(
                self._run_view,
                prepared[id(entry[0])],
                entry,
                frame.shape,
            )
            for entry in self.views
        ]

    def finish(self, futures: list[_Future]) -> list[dict[str, Any]]:
        groups = [future.result() for future in futures]
        merged = [item for group in groups for item in group]
        for order, item in enumerate(merged):
            item["_nms_order"] = order
        selected = nms_records(
            merged,
            float(self.config["scan"]["global_nms_iou"]),
            float(self.config["scan"]["containment_threshold"]),
        )
        for item in selected:
            item.pop("_nms_order", None)
        return selected

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)


__all__ = ["ScanRunner"]
