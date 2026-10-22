"""Object-local sparse LK flow on a normalized, virtually padded ROI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .tracker import (
    BoxKalman,
    _estimate_affine_endpoint_flow,
    _estimate_flow,
)


@dataclass
class TrackingROI:
    gray: np.ndarray
    origin_x: int
    origin_y: int
    side: int
    output_size: int

    @property
    def scale(self) -> float:
        return self.output_size / self.side

    def local_box(self, value: np.ndarray) -> np.ndarray:
        return (value - np.asarray([self.origin_x, self.origin_y, self.origin_x, self.origin_y])) * self.scale

    def global_box(self, value: np.ndarray) -> np.ndarray:
        return value / self.scale + np.asarray([self.origin_x, self.origin_y, self.origin_x, self.origin_y])


def _crop_canvas(frame: np.ndarray, origin_x: int, origin_y: int, side: int) -> np.ndarray:
    canvas = np.zeros((side, side, 3), dtype=frame.dtype)
    source_x1, source_y1 = max(0, origin_x), max(0, origin_y)
    source_x2 = min(frame.shape[1], origin_x + side)
    source_y2 = min(frame.shape[0], origin_y + side)
    if source_x2 > source_x1 and source_y2 > source_y1:
        canvas[
            source_y1 - origin_y : source_y2 - origin_y,
            source_x1 - origin_x : source_x2 - origin_x,
        ] = frame[source_y1:source_y2, source_x1:source_x2]
    return canvas


def tracking_roi(
    frame: np.ndarray,
    value: np.ndarray,
    *,
    expansion: float,
    output_size: int,
    transform: TrackingROI | None = None,
    max_source_canvas_side: int | None = None,
) -> TrackingROI:
    if transform is None:
        center = (value[:2] + value[2:]) * 0.5
        side = max(16, math.ceil(max(value[2] - value[0], value[3] - value[1]) * expansion))
        origin_x = math.floor(center[0] - side * 0.5)
        origin_y = math.floor(center[1] - side * 0.5)
    else:
        side, origin_x, origin_y = transform.side, transform.origin_x, transform.origin_y
    if max_source_canvas_side is not None and side > max_source_canvas_side:
        # Avoid allocating a multi-thousand-pixel intermediate canvas for a
        # large face. Map the same virtual, zero-padded source square directly
        # into the normalized tracker input.
        scale = output_size / side
        transform_matrix = np.asarray(
            [[scale, 0.0, -origin_x * scale], [0.0, scale, -origin_y * scale]],
            dtype=np.float64,
        )
        canvas = cv2.warpAffine(
            frame,
            transform_matrix,
            (output_size, output_size),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    else:
        canvas = cv2.resize(
            _crop_canvas(frame, origin_x, origin_y, side),
            (output_size, output_size),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY,
    )
    return TrackingROI(gray, origin_x, origin_y, side, output_size)


class AffineEndpointState:
    """Independent short-horizon affine state that never mutates main LK."""

    def __init__(
        self,
        frame: np.ndarray,
        value: list[float] | np.ndarray,
        settings: dict[str, Any],
    ):
        self.settings = settings
        self.box = np.asarray(value, dtype=np.float64)
        self.roi = tracking_roi(
            frame,
            self.box,
            expansion=float(settings["roi_expansion"]),
            output_size=int(settings["roi_size"]),
            max_source_canvas_side=int(settings.get("max_source_canvas_side_pixels", 1024)),
        )

    def step(self, frame: np.ndarray) -> dict[str, Any]:
        current = tracking_roi(
            frame,
            self.box,
            expansion=float(self.settings["roi_expansion"]),
            output_size=int(self.settings["roi_size"]),
            transform=self.roi,
            max_source_canvas_side=int(self.settings.get("max_source_canvas_side_pixels", 1024)),
        )
        local_settings = dict(self.settings)
        local_settings["padding_pixels"] = int(self.settings.get("roi_edge_padding_pixels", 12))
        result = _estimate_affine_endpoint_flow(
            self.roi.gray,
            current.gray,
            self.roi.local_box(self.box),
            local_settings,
        )
        if not bool(result["valid"]):
            return {**result, "box": self.box.copy()}
        self.box = self.roi.global_box(np.asarray(result["box"], dtype=np.float64))
        self.roi = tracking_roi(
            frame,
            self.box,
            expansion=float(self.settings["roi_expansion"]),
            output_size=int(self.settings["roi_size"]),
            max_source_canvas_side=int(self.settings.get("max_source_canvas_side_pixels", 1024)),
        )
        return {**result, "box": self.box.copy()}


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


class ROIFlowState:
    def __init__(self, frame: np.ndarray, value: list[float] | np.ndarray, settings: dict[str, Any]):
        self.settings = settings
        self.box = np.asarray(value, dtype=np.float64)
        # ``box`` is the last trusted/Kalman state. ``search_box`` and ``roi``
        # may advance provisionally during coast so LK can recover after a
        # short occlusion. A provisional recovery is promoted only if a direct
        # reverse flow maps it back to ``trusted_roi``/``trusted_box``.
        self.trusted_box = self.box.copy()
        self.search_box = self.box.copy()
        self.kalman = BoxKalman(self.box, settings)
        self.roi = tracking_roi(
            frame,
            self.box,
            expansion=float(settings["roi_expansion"]),
            output_size=int(settings["roi_size"]),
            max_source_canvas_side=int(settings.get("max_source_canvas_side_pixels", 1024)),
        )
        self.trusted_roi = TrackingROI(
            self.roi.gray,
            self.roi.origin_x,
            self.roi.origin_y,
            self.roi.side,
            self.roi.output_size,
        )
        self.coast = 0

    def rebase(self, offset_x: int, offset_y: int) -> None:
        """Express the same state in a frame crop with a different origin."""

        offset = np.asarray([offset_x, offset_y, offset_x, offset_y], dtype=np.float64)
        self.box += offset
        self.trusted_box += offset
        self.search_box += offset
        self.kalman.cx.position += float(offset_x)
        self.kalman.cy.position += float(offset_y)
        self.roi.origin_x += int(offset_x)
        self.roi.origin_y += int(offset_y)
        self.trusted_roi.origin_x += int(offset_x)
        self.trusted_roi.origin_y += int(offset_y)

    def step(self, frame: np.ndarray) -> dict[str, Any]:
        if not bool(self.settings.get("flow_updates_size", True)):
            # LK scale is useful as a short-lived motion cue but drifts when the
            # tracked region contains hair, hands, or torso texture.  Keep size
            # state measurement-driven and let flow update translation only.
            self.kalman.lw.velocity = 0.0
            self.kalman.lh.velocity = 0.0
        self.kalman.predict()
        current = tracking_roi(
            frame,
            self.search_box,
            expansion=float(self.settings["roi_expansion"]),
            output_size=int(self.settings["roi_size"]),
            transform=self.roi,
            max_source_canvas_side=int(self.settings.get("max_source_canvas_side_pixels", 1024)),
        )
        local_settings = dict(self.settings)
        local_settings["padding_pixels"] = int(self.settings.get("roi_edge_padding_pixels", 12))
        flow = _estimate_flow(
            self.roi.gray,
            current.gray,
            self.roi.local_box(self.search_box),
            local_settings,
        )
        prior_coast = self.coast
        lk_valid = bool(flow["valid"])
        raw_valid = lk_valid
        measurement = self.roi.global_box(np.asarray(flow["box"], dtype=np.float64)) if raw_valid else None
        require_cycle = bool(self.settings.get("require_cycle_consistency_after_coast", False))
        cycle_checked = bool(raw_valid and prior_coast > 0 and require_cycle)
        cycle_valid = not cycle_checked
        cycle_iou: float | None = None
        if cycle_checked:
            assert measurement is not None
            if prior_coast <= int(self.settings.get("max_coast_frames", 0)):
                current_in_trusted = tracking_roi(
                    frame,
                    measurement,
                    expansion=float(self.settings["roi_expansion"]),
                    output_size=int(self.settings["roi_size"]),
                    transform=self.trusted_roi,
                    max_source_canvas_side=int(self.settings.get("max_source_canvas_side_pixels", 1024)),
                )
                reverse_flow = _estimate_flow(
                    current_in_trusted.gray,
                    self.trusted_roi.gray,
                    current_in_trusted.local_box(measurement),
                    local_settings,
                )
                if reverse_flow["valid"]:
                    reverse_box = self.trusted_roi.global_box(
                        np.asarray(reverse_flow["box"], dtype=np.float64)
                    )
                    cycle_iou = _box_iou(reverse_box, self.trusted_box)
                    cycle_valid = cycle_iou >= float(self.settings.get("recovery_cycle_min_iou", 0.10))
        measurement_valid = bool(raw_valid and cycle_valid)
        trusted_valid = measurement_valid
        if measurement_valid:
            assert measurement is not None
            quality = max(0.05, float(flow["quality"]))
            self.kalman.update(
                measurement,
                float(self.settings["flow_measurement_noise"]) / quality,
                0.05 / quality,
                update_size=bool(self.settings.get("flow_updates_size", True)),
            )
            self.coast = 0
        else:
            self.coast += 1
        self.box = self.kalman.box()
        if trusted_valid:
            self.trusted_box = self.box.copy()
            self.search_box = self.box.copy()
            self.roi = tracking_roi(
                frame,
                self.box,
                expansion=float(self.settings["roi_expansion"]),
                output_size=int(self.settings["roi_size"]),
                max_source_canvas_side=int(self.settings.get("max_source_canvas_side_pixels", 1024)),
            )
            self.trusted_roi = TrackingROI(
                self.roi.gray,
                self.roi.origin_x,
                self.roi.origin_y,
                self.roi.side,
                self.roi.output_size,
            )
        else:
            # Continue searching on the provisional measurement, but keep the
            # public/Kalman box and trusted reference unchanged. This permits
            # recovery without allowing unrelated texture to become trusted.
            self.search_box = (
                np.asarray(measurement, dtype=np.float64)
                if raw_valid and measurement is not None
                else self.box.copy()
            )
            self.roi = tracking_roi(
                frame,
                self.search_box,
                expansion=float(self.settings["roi_expansion"]),
                output_size=int(self.settings["roi_size"]),
                max_source_canvas_side=int(self.settings.get("max_source_canvas_side_pixels", 1024)),
            )
        return {
            **flow,
            "valid": trusted_valid,
            "lk_valid": lk_valid,
            "flow_measurement_valid": measurement_valid,
            "raw_valid": raw_valid,
            "box": self.box.copy(),
            "provisional_box": self.search_box.copy(),
            "coast": self.coast,
            "cycle_checked": cycle_checked,
            "cycle_valid": cycle_valid,
            "cycle_iou": cycle_iou,
            "recovered_from_coast": bool(trusted_valid and prior_coast > 0),
        }

    def correct(
        self,
        frame: np.ndarray,
        value: list[float] | np.ndarray,
        *,
        update_size: bool = True,
    ) -> None:
        measurement = np.asarray(value, dtype=np.float64)
        self.kalman.update(
            measurement,
            float(self.settings["detector_measurement_noise"]),
            0.02,
            update_size=update_size,
        )
        self.kalman.lw.velocity = 0.0
        self.kalman.lh.velocity = 0.0
        self.box = self.kalman.box()
        self.trusted_box = self.box.copy()
        self.search_box = self.box.copy()
        self.coast = 0
        self.roi = tracking_roi(
            frame,
            self.box,
            expansion=float(self.settings["roi_expansion"]),
            output_size=int(self.settings["roi_size"]),
            max_source_canvas_side=int(self.settings.get("max_source_canvas_side_pixels", 1024)),
        )
        self.trusted_roi = TrackingROI(
            self.roi.gray,
            self.roi.origin_x,
            self.roi.origin_y,
            self.roi.side,
            self.roi.output_size,
        )


__all__ = [
    "AffineEndpointState",
    "ROIFlowState",
    "TrackingROI",
    "tracking_roi",
]
