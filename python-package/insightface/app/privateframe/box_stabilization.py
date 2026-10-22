"""Robust, track-local output box smoothing for video face redaction."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise
from typing import Any

import numpy as np


def _state(value: list[float]) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64)
    size = np.maximum(2.0, box[2:] - box[:2])
    center = (box[:2] + box[2:]) * 0.5
    return np.asarray(
        [center[0], center[1], math.log(size[0]), math.log(size[1])],
        dtype=np.float64,
    )


def _box(value: np.ndarray) -> np.ndarray:
    size = np.exp(value[2:])
    return np.concatenate((value[:2] - size * 0.5, value[:2] + size * 0.5))


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    output = np.empty_like(values)
    for index in range(len(values)):
        first = max(0, index - radius)
        last = min(len(values), index + radius + 1)
        output[index] = np.median(values[first:last], axis=0)
    return output


def _clip_outliers(
    values: np.ndarray,
    baseline: np.ndarray,
    max_center_innovation: float,
    max_size_ratio: float,
) -> np.ndarray:
    output = values.copy()
    maximum_log_size = math.log(max_size_ratio)
    for index, (raw, median) in enumerate(zip(values, baseline)):
        scale = math.sqrt(math.exp(median[2] + median[3]))
        center = raw[:2] - median[:2]
        length = float(np.linalg.norm(center))
        maximum_center = max_center_innovation * max(2.0, scale)
        if length > maximum_center:
            center *= maximum_center / max(length, 1e-9)
        output[index, :2] = median[:2] + center
        output[index, 2:] = median[2:] + np.clip(
            raw[2:] - median[2:], -maximum_log_size, maximum_log_size
        )
    return output


def _bidirectional(values: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    forward = values.copy()
    backward = values.copy()
    for index in range(1, len(values)):
        forward[index] = forward[index - 1] + alpha * (
            values[index] - forward[index - 1]
        )
    for index in range(len(values) - 2, -1, -1):
        backward[index] = backward[index + 1] + alpha * (
            values[index] - backward[index + 1]
        )
    return (forward + backward) * 0.5


def _regularize_between_detector_anchors(
    segment: list[dict[str, Any]],
    robust: np.ndarray,
    filtered: np.ndarray,
    strength: float,
    minimum_gap_frames: int,
    maximum_gap_frames: int,
) -> np.ndarray:
    """Constrain medium detector gaps without trusting every detector frame.

    Optical flow and low-confidence local SCRFD boxes can drift together for
    several frames.  When the same accepted trajectory has detector evidence
    on both sides, the two detector boxes provide a stronger zero-phase
    corridor.  Very short gaps are intentionally ignored: directly following
    every adjacent detector box would reintroduce detector jitter.
    """

    if strength <= 0.0:
        return filtered
    anchors = [
        index
        for index, item in enumerate(segment)
        if item.get("source") == "detector"
    ]
    output = filtered.copy()
    for first, last in pairwise(anchors):
        first_frame = int(segment[first]["frame_idx"])
        last_frame = int(segment[last]["frame_idx"])
        gap = last_frame - first_frame
        if gap < minimum_gap_frames or gap > maximum_gap_frames:
            continue
        for index in range(first, last + 1):
            weight = (
                int(segment[index]["frame_idx"]) - first_frame
            ) / max(1, gap)
            corridor = robust[first] * (1.0 - weight) + robust[last] * weight
            output[index] = output[index] * (1.0 - strength) + corridor * strength
    return output


def _scale(value: list[float]) -> float:
    box = np.asarray(value, dtype=np.float64)
    size = np.maximum(2.0, box[2:] - box[:2])
    return math.sqrt(float(size[0] * size[1]))


def _symmetric_ratio(first: float, second: float) -> float:
    return max(first, second) / max(1e-9, min(first, second))


def _persistent_scale_boundaries(
    values: list[dict[str, Any]],
    settings: dict[str, Any],
    scene_mean_absdiff_by_frame: dict[int, float] | None = None,
) -> list[int]:
    """Find sustained detector-backed size steps without reacting to one outlier."""

    window = int(settings["window_frames"])
    minimum_detector_frames = int(settings["min_detector_frames_per_side"])
    minimum_instantaneous = float(settings["min_instantaneous_scale_ratio"])
    minimum_persistent = float(settings["min_persistent_scale_ratio"])
    maximum_variation = float(settings["max_within_regime_ratio"])
    minimum_scene_difference = float(settings["min_scene_mean_absdiff"])
    if len(values) < window * 2:
        return []
    scales = np.asarray([_scale(item["box"]) for item in values], dtype=np.float64)
    boundaries: list[int] = []
    for boundary in range(window, len(values) - window + 1):
        frame_idx = int(values[boundary]["frame_idx"])
        if (
            scene_mean_absdiff_by_frame is None
            or float(scene_mean_absdiff_by_frame.get(frame_idx, 0.0))
            < minimum_scene_difference
        ):
            continue
        left_items = values[boundary - window : boundary]
        right_items = values[boundary : boundary + window]
        if (
            sum(item.get("source") == "detector" for item in left_items)
            < minimum_detector_frames
        ):
            continue
        if (
            sum(item.get("source") == "detector" for item in right_items)
            < minimum_detector_frames
        ):
            continue
        if (
            _symmetric_ratio(scales[boundary - 1], scales[boundary])
            < minimum_instantaneous
        ):
            continue
        left = scales[boundary - window : boundary]
        right = scales[boundary : boundary + window]
        left_median = float(np.median(left))
        right_median = float(np.median(right))
        if _symmetric_ratio(left_median, right_median) < minimum_persistent:
            continue
        if any(
            _symmetric_ratio(float(value), left_median) > maximum_variation
            for value in left
        ):
            continue
        if any(
            _symmetric_ratio(float(value), right_median) > maximum_variation
            for value in right
        ):
            continue
        boundaries.append(boundary)
    return boundaries


def _segments(
    values: list[dict[str, Any]],
    reset_gap_frames: int,
    scene_cut_frames: set[int],
    scene_mean_absdiff_by_frame: dict[int, float] | None,
    change_point_settings: dict[str, Any] | None,
) -> list[list[dict[str, Any]]]:
    contiguous: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in sorted(values, key=lambda value: int(value["frame_idx"])):
        frame_idx = int(item["frame_idx"])
        if current and (
            frame_idx - int(current[-1]["frame_idx"]) > reset_gap_frames
            or frame_idx in scene_cut_frames
        ):
            contiguous.append(current)
            current = []
        current.append(item)
    if current:
        contiguous.append(current)

    if not change_point_settings or not bool(
        change_point_settings.get("enabled", False)
    ):
        return contiguous
    output: list[list[dict[str, Any]]] = []
    for segment in contiguous:
        boundaries = _persistent_scale_boundaries(
            segment,
            change_point_settings,
            scene_mean_absdiff_by_frame,
        )
        first = 0
        for boundary in boundaries:
            output.append(segment[first:boundary])
            first = boundary
        output.append(segment[first:])
    return [segment for segment in output if segment]


def stabilize_observations(
    observations: list[dict[str, Any]],
    settings: dict[str, Any] | None,
    *,
    scene_cut_frames: set[int] | None = None,
    scene_mean_absdiff_by_frame: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Return observations with robust zero-phase box stabilization applied.

    Track ids, frame indices, sources and observation count are immutable. The
    operation runs only after admission/finalization, so it cannot create,
    delete or accept a trajectory.
    """

    copied = [dict(item) for item in observations]
    if not settings or not bool(settings.get("enabled", False)):
        return copied
    window = int(settings["median_window"])
    if window < 3 or window % 2 == 0:
        raise ValueError("box_stabilization.median_window must be odd and at least 3")
    alpha = np.asarray(
        [
            float(settings["center_alpha"]),
            float(settings["center_alpha"]),
            float(settings["size_alpha"]),
            float(settings["size_alpha"]),
        ],
        dtype=np.float64,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in copied:
        grouped[str(item["track_id"])].append(item)
    for values in grouped.values():
        for segment in _segments(
            values,
            int(settings["reset_gap_frames"]),
            scene_cut_frames or set(),
            scene_mean_absdiff_by_frame,
            settings.get("change_point_reset"),
        ):
            if len(segment) < int(settings["min_segment_frames"]):
                continue
            raw = np.stack([_state(item["box"]) for item in segment])
            median = _rolling_median(raw, window)
            robust = _clip_outliers(
                raw,
                median,
                float(settings["max_center_innovation"]),
                float(settings["max_size_ratio"]),
            )
            filtered = _bidirectional(robust, alpha)
            filtered = _regularize_between_detector_anchors(
                segment,
                robust,
                filtered,
                float(settings.get("detector_anchor_strength", 0.0)),
                int(settings.get("detector_anchor_min_gap_frames", 1)),
                int(settings.get("detector_anchor_max_gap_frames", 1)),
            )
            if float(settings.get("detector_anchor_strength", 0.0)) > 0.0:
                # Anchor corridors are piecewise by construction.  A second
                # zero-phase pass removes boundary discontinuities without
                # adding temporal lag or weakening a sustained correction.
                filtered = _bidirectional(filtered, alpha)
            for item, original, value in zip(segment, raw, filtered):
                item["raw_output_box"] = _box(original).tolist()
                item["box"] = _box(value).tolist()
                item["motion_box"] = item["box"]
                item["box_stabilization"] = "robust_bidirectional"
    return sorted(copied, key=lambda item: (int(item["frame_idx"]), item["track_id"]))


__all__ = ["stabilize_observations"]
