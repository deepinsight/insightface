"""Kalman and sparse OpenCV LK primitives used by the streaming tracker."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .geometry import clip


class ScalarKalman:
    def __init__(self, position: float):
        self.position = position
        self.velocity = 0.0
        self.p00, self.p01, self.p10, self.p11 = 4.0, 0.0, 0.0, 16.0

    def predict(self, position_noise: float, velocity_noise: float) -> None:
        self.position += self.velocity
        self.p00, self.p01, self.p10, self.p11 = (
            self.p00 + self.p01 + self.p10 + self.p11 + position_noise,
            self.p01 + self.p11,
            self.p10 + self.p11,
            self.p11 + velocity_noise,
        )

    def update(self, measurement: float, variance: float) -> None:
        denominator = self.p00 + variance
        k0, k1 = self.p00 / denominator, self.p10 / denominator
        innovation = measurement - self.position
        self.position += k0 * innovation
        self.velocity += k1 * innovation
        old00, old01 = self.p00, self.p01
        self.p00 = (1.0 - k0) * old00
        self.p01 = (1.0 - k0) * old01
        self.p10 -= k1 * old00
        self.p11 -= k1 * old01


class BoxKalman:
    def __init__(self, value: np.ndarray, settings: dict[str, Any]):
        self.settings = settings
        center = (value[:2] + value[2:]) * 0.5
        self.cx, self.cy = ScalarKalman(float(center[0])), ScalarKalman(float(center[1]))
        self.lw = ScalarKalman(math.log(max(2.0, float(value[2] - value[0]))))
        self.lh = ScalarKalman(math.log(max(2.0, float(value[3] - value[1]))))

    def predict(self) -> None:
        position = float(self.settings["process_noise_position"])
        velocity = float(self.settings["process_noise_velocity"])
        self.cx.predict(position, velocity)
        self.cy.predict(position, velocity)
        self.lw.predict(position * 0.002, velocity * 0.0005)
        self.lh.predict(position * 0.002, velocity * 0.0005)

    def update(
        self, value: np.ndarray, center_sigma: float, size_sigma: float, update_size: bool = True
    ) -> None:
        c = (value[:2] + value[2:]) * 0.5
        self.cx.update(float(c[0]), center_sigma * center_sigma)
        self.cy.update(float(c[1]), center_sigma * center_sigma)
        if update_size:
            self.lw.update(math.log(max(2.0, float(value[2] - value[0]))), size_sigma * size_sigma)
            self.lh.update(math.log(max(2.0, float(value[3] - value[1]))), size_sigma * size_sigma)

    def box(self) -> np.ndarray:
        width, height = math.exp(self.lw.position), math.exp(self.lh.position)
        return np.asarray(
            [
                self.cx.position - width * 0.5,
                self.cy.position - height * 0.5,
                self.cx.position + width * 0.5,
                self.cy.position + height * 0.5,
            ],
            dtype=np.float64,
        )


def _select_points(gray: np.ndarray, value: np.ndarray, settings: dict[str, Any]) -> np.ndarray | None:
    expansion = float(settings["feature_box_expansion"])
    width, height = value[2] - value[0], value[3] - value[1]
    target = clip(
        [
            value[0] - width * expansion,
            value[1] - height * expansion,
            value[2] + width * expansion,
            value[3] + height * expansion,
        ],
        gray.shape[1],
        gray.shape[0],
    )
    x1, y1, x2, y2 = math.floor(target[0]), math.floor(target[1]), math.ceil(target[2]), math.ceil(target[3])
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    mask = np.zeros_like(gray)
    mask[max(0, y1) : min(gray.shape[0], y2), max(0, x1) : min(gray.shape[1], x2)] = 255
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(settings["max_points"]),
        qualityLevel=0.01,
        minDistance=max(2.0, min(width, height) / math.sqrt(max(1, int(settings["max_points"])))),
        mask=mask,
        blockSize=3,
        useHarrisDetector=False,
    )


def _estimate_flow(
    previous: np.ndarray,
    current: np.ndarray,
    prior_box: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    points = _select_points(previous, prior_box, settings)
    selected = 0 if points is None else len(points)

    def invalid(inliers: int = 0) -> dict[str, Any]:
        return {
            "valid": False,
            "selected": selected,
            "inliers": inliers,
            "inlier_fraction": inliers / selected if selected else 0.0,
            "quality": 0.0,
        }

    if points is None or selected < int(settings["min_points"]):
        return invalid()
    levels = max(0, int(settings["pyramid_levels"]) - 1)
    lk = {
        "winSize": (2 * int(settings["window_radius"]) + 1,) * 2,
        "maxLevel": levels,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(settings["max_iterations"]),
            float(settings["termination_epsilon"]),
        ),
        "minEigThreshold": 1e-4,
    }
    forward, status_forward, _errors = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **lk)
    if forward is None:
        return invalid()
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **lk)
    if backward is None:
        return invalid()
    p0, p1, pback = points[:, 0], forward[:, 0], backward[:, 0]
    fb = np.linalg.norm(pback - p0, axis=1)
    valid = (status_forward[:, 0] > 0) & (status_backward[:, 0] > 0) & np.isfinite(fb)
    valid &= fb <= float(settings["forward_backward_max_error"])
    valid &= (p1[:, 0] >= 0) & (p1[:, 0] < current.shape[1]) & (p1[:, 1] >= 0) & (p1[:, 1] < current.shape[0])
    p0, p1, fb = p0[valid], p1[valid], fb[valid]
    if len(p0) < int(settings["min_points"]):
        return invalid(len(p0))
    before_center, after_center = np.median(p0, axis=0), np.median(p1, axis=0)
    before_radius = np.linalg.norm(p0 - before_center, axis=1)
    after_radius = np.linalg.norm(p1 - after_center, axis=1)
    minimum_radius = max(2.0, 0.08 * min(prior_box[2] - prior_box[0], prior_box[3] - prior_box[1]))
    scale_values = (
        after_radius[before_radius >= minimum_radius] / before_radius[before_radius >= minimum_radius]
    )
    scale = float(np.median(scale_values)) if len(scale_values) >= 3 else 1.0
    scale = float(
        np.clip(scale, 1.0 - float(settings["max_scale_change"]), 1.0 + float(settings["max_scale_change"]))
    )
    edge = int(settings["padding_pixels"])
    if edge > 0 and (
        prior_box[0] < edge
        or prior_box[1] < edge
        or prior_box[2] > current.shape[1] - edge
        or prior_box[3] > current.shape[0] - edge
    ):
        scale = 1.0
    expected = after_center + scale * (p0 - before_center)
    residual = np.linalg.norm(p1 - expected, axis=1)
    inlier = residual <= float(settings["residual_threshold"])
    p0, p1, fb, residual = p0[inlier], p1[inlier], fb[inlier], residual[inlier]
    if len(p0) < int(settings["min_points"]):
        return invalid(len(p0))
    before_center, after_center = np.median(p0, axis=0), np.median(p1, axis=0)
    displacement = after_center - before_center
    c = (prior_box[:2] + prior_box[2:]) * 0.5 + displacement
    width = max(2.0, (prior_box[2] - prior_box[0]) * scale)
    height = max(2.0, (prior_box[3] - prior_box[1]) * scale)
    output_box = np.asarray(
        [c[0] - width * 0.5, c[1] - height * 0.5, c[0] + width * 0.5, c[1] + height * 0.5]
    )
    quality = (
        len(p0)
        / selected
        * math.exp(-float(np.median(fb)) / float(settings["forward_backward_max_error"]))
        * math.exp(-float(np.median(residual)) / float(settings["residual_threshold"]))
    )
    return {
        "valid": True,
        "box": output_box,
        "selected": selected,
        "inliers": len(p0),
        "inlier_fraction": len(p0) / selected,
        "quality": quality,
    }


def _estimate_affine_endpoint_flow(
    previous: np.ndarray,
    current: np.ndarray,
    prior_box: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Strict partial-affine LK used only by isolated endpoint repair."""

    def invalid(
        selected: int,
        inliers: int = 0,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "valid": False,
            "selected": selected,
            "inliers": inliers,
            "inlier_fraction": inliers / selected if selected else 0.0,
            "quality": 0.0,
            **extra,
        }

    points = _select_points(previous, prior_box, settings)
    selected = 0 if points is None else len(points)
    minimum = int(settings["min_points"])
    if points is None or selected < minimum:
        return invalid(selected)

    # These are fixed implementation constants rather than user-facing
    # tuning. They are used only after the normal small-window LK endpoint has
    # already failed.
    lk = {
        "winSize": (15, 15),
        "maxLevel": max(0, int(settings["pyramid_levels"]) - 1),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            float(settings["termination_epsilon"]),
        ),
        "minEigThreshold": 1e-4,
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **lk)
    if forward is None:
        return invalid(selected)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **lk)
    if backward is None:
        return invalid(selected)

    p0 = points[:, 0]
    p1 = forward[:, 0]
    pback = backward[:, 0]
    fb = np.linalg.norm(pback - p0, axis=1)
    valid = (
        (forward_status[:, 0] > 0)
        & (backward_status[:, 0] > 0)
        & np.isfinite(fb)
        & (fb <= float(settings["forward_backward_max_error"]))
        & (p1[:, 0] >= 0)
        & (p1[:, 0] < current.shape[1])
        & (p1[:, 1] >= 0)
        & (p1[:, 1] < current.shape[0])
    )
    p0, p1, fb = p0[valid], p1[valid], fb[valid]
    if len(p0) < minimum:
        return invalid(selected, len(p0))

    matrix, mask = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(settings["residual_threshold"]),
        maxIters=500,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or mask is None:
        return invalid(selected)
    affine_inliers = mask[:, 0] > 0
    p0, p1, fb = p0[affine_inliers], p1[affine_inliers], fb[affine_inliers]
    if len(p0) < minimum:
        return invalid(selected, len(p0))

    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    rotation = abs(math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))))
    maximum_scale_change = float(settings["max_scale_change"])
    if not 1.0 - maximum_scale_change <= scale <= 1.0 + maximum_scale_change or rotation > 15.0:
        return invalid(
            selected,
            len(p0),
            affine_scale=scale,
            affine_rotation_degrees=rotation,
        )

    normalized = (p0 - prior_box[:2]) / np.maximum(prior_box[2:] - prior_box[:2], 1.0)
    quadrants = {(int(point[0] >= 0.5), int(point[1] >= 0.5)) for point in normalized}
    span = np.ptp(p0, axis=0) / np.maximum(prior_box[2:] - prior_box[:2], 1.0)
    spread = float(math.sqrt(max(0.0, float(span[0] * span[1]))))
    if len(quadrants) < 2 or spread < 0.15:
        return invalid(
            selected,
            len(p0),
            occupied_quadrants=len(quadrants),
            feature_spread=spread,
            affine_scale=scale,
            affine_rotation_degrees=rotation,
        )

    predicted = cv2.transform(p0[None, :, :], matrix)[0]
    residual = np.linalg.norm(p1 - predicted, axis=1)
    corners = np.asarray(
        [
            [prior_box[0], prior_box[1]],
            [prior_box[2], prior_box[1]],
            [prior_box[2], prior_box[3]],
            [prior_box[0], prior_box[3]],
        ],
        dtype=np.float64,
    )
    transformed = cv2.transform(corners[None, :, :], matrix)[0]
    output_box = np.asarray(
        [
            transformed[:, 0].min(),
            transformed[:, 1].min(),
            transformed[:, 0].max(),
            transformed[:, 1].max(),
        ],
        dtype=np.float64,
    )
    quality = (
        len(p0)
        / selected
        * math.exp(-float(np.median(fb)) / float(settings["forward_backward_max_error"]))
        * math.exp(-float(np.median(residual)) / float(settings["residual_threshold"]))
    )
    return {
        "valid": True,
        "box": output_box,
        "selected": selected,
        "inliers": len(p0),
        "inlier_fraction": len(p0) / selected,
        "quality": quality,
        "occupied_quadrants": len(quadrants),
        "feature_spread": spread,
        "affine_scale": scale,
        "affine_rotation_degrees": rotation,
    }


def _rank(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(item["center_distance"]),
        abs(math.log(max(float(item["area_ratio"]), 1e-12))),
        0 if int(item["direction"]) > 0 else 1,
        -float(item["quality"]),
    )


def _select_track_frame_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select one motion candidate for each track and frame.

    Cross-track suppression deliberately does not belong here: these boxes
    have not passed admission or output stabilization yet.  Comparing tracks
    at this stage made a later box movement capable of invalidating the
    geometry that justified an early suppression.
    """

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in candidates:
        grouped.setdefault((item["track_id"], int(item["frame_idx"])), []).append(item)
    selected_candidates = []
    for values in grouped.values():
        selected = dict(min(values, key=_rank))
        forward = next((item for item in values if int(item["direction"]) > 0), None)
        reverse = next((item for item in values if int(item["direction"]) < 0), None)
        if (
            forward
            and reverse
            and int(forward["anchor_frame"]) < int(reverse["anchor_frame"])
            # A local SCRFD match is a direct geometry measurement.  Do not
            # replace it with the less precise interpolation of two motion
            # boxes during forward/reverse de-duplication.
            and not str(selected.get("geometry_source", "")).startswith("local_scrfd")
        ):
            fraction = (int(selected["frame_idx"]) - int(forward["anchor_frame"])) / max(
                1, int(reverse["anchor_frame"]) - int(forward["anchor_frame"])
            )
            selected["motion_box"] = (
                np.asarray(forward["motion_box"]) * (1.0 - fraction) + np.asarray(reverse["motion_box"]) * fraction
            ).tolist()
        selected_candidates.append(selected)
    return sorted(
        selected_candidates,
        key=lambda item: (int(item["frame_idx"]), str(item["track_id"])),
    )


__all__ = ["BoxKalman"]
