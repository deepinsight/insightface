"""Small, NumPy-friendly geometry primitives used by the streaming pipeline."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def box(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(4)


def area(value: Any) -> float:
    b = box(value)
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def intersection(first: Any, second: Any) -> float:
    a, b = box(first), box(second)
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def iou(first: Any, second: Any) -> float:
    overlap = intersection(first, second)
    return overlap / max(area(first) + area(second) - overlap, 1e-9)


def containment(first: Any, second: Any) -> float:
    return intersection(first, second) / max(min(area(first), area(second)), 1e-9)


def area_ratio(first: Any, second: Any) -> float:
    a, b = max(area(first), 1e-9), max(area(second), 1e-9)
    return max(a / b, b / a)


def covers_reference(
    reference: Any,
    candidate: Any,
    *,
    min_coverage: float,
    max_candidate_area_ratio: float,
) -> bool:
    """Return whether one candidate safely covers a reference geometry.

    Unlike symmetric IoU/containment NMS, this privacy check does not let a
    smaller candidate suppress an accepted reference merely because the
    smaller box is fully contained by it.
    """

    reference_area = area(reference)
    if reference_area <= 0.0:
        return False
    coverage = intersection(reference, candidate) / reference_area
    candidate_area_ratio = area(candidate) / reference_area
    return (
        coverage >= float(min_coverage)
        and candidate_area_ratio <= float(max_candidate_area_ratio)
    )


def mutually_covers(
    first: Any,
    second: Any,
    *,
    min_coverage: float,
    max_area_ratio: float,
) -> bool:
    """Return whether either box can safely stand in for the other."""

    return covers_reference(
        first,
        second,
        min_coverage=min_coverage,
        max_candidate_area_ratio=max_area_ratio,
    ) and covers_reference(
        second,
        first,
        min_coverage=min_coverage,
        max_candidate_area_ratio=max_area_ratio,
    )


def center(value: Any) -> np.ndarray:
    b = box(value)
    return (b[:2] + b[2:]) * 0.5


def normalized_center_distance(first: Any, second: Any) -> float:
    scale = math.sqrt(max(min(area(first), area(second)), 1.0))
    return float(np.linalg.norm(center(first) - center(second)) / scale)


def clip(value: Any, width: int, height: int) -> np.ndarray:
    b = box(value).copy()
    b[[0, 2]] = np.clip(b[[0, 2]], 0.0, float(width))
    b[[1, 3]] = np.clip(b[[1, 3]], 0.0, float(height))
    return b


def inverse_cardinal_points(
    value: Any,
    angle: int,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """Map pixel-center points back through a cardinal image rotation.

    Boxes in this package use continuous half-open boundaries and therefore
    transform with ``width``/``height``. Landmark coordinates identify pixel
    centers, matching OpenCV's rotation mapping, so their inverse uses
    ``width - 1``/``height - 1``.
    """

    points = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if angle == 0:
        return points.copy()
    result = np.empty_like(points)
    if angle == 90:
        result[:, 0] = float(original_width - 1) - points[:, 1]
        result[:, 1] = points[:, 0]
        return result
    if angle == -90:
        result[:, 0] = points[:, 1]
        result[:, 1] = float(original_height - 1) - points[:, 0]
        return result
    raise ValueError(f"unsupported cardinal angle {angle}")


def median(values: Iterable[float]) -> float | None:
    data = list(values)
    return float(np.median(data)) if data else None


def nms_records(
    records: list[dict[str, Any]], iou_threshold: float, containment_threshold: float
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            str(item.get("detection_id", "")),
            int(item.get("_nms_order", 0)),
        ),
    )
    kept: list[dict[str, Any]] = []
    for item in ordered:
        if any(
            iou(item["box"], prior["box"]) >= iou_threshold
            or containment(item["box"], prior["box"]) >= containment_threshold
            for prior in kept
        ):
            continue
        kept.append(item)
    return kept


def continuity_segments(
    values: list[dict[str, Any]], max_center_jump: float, max_area_ratio: float
) -> list[list[dict[str, Any]]]:
    if not values:
        return []
    result: list[list[dict[str, Any]]] = [[values[0]]]
    for item in values[1:]:
        prior = result[-1][-1]
        if (
            int(item["frame_idx"]) != int(prior["frame_idx"]) + 1
            or normalized_center_distance(item["box"], prior["box"]) > max_center_jump
            or area_ratio(item["box"], prior["box"]) > max_area_ratio
        ):
            result.append([])
        result[-1].append(item)
    return result


__all__ = [
    "area", "area_ratio", "box", "center", "clip", "containment", "continuity_segments",
    "covers_reference", "intersection", "inverse_cardinal_points", "iou", "median", "mutually_covers", "nms_records",
    "normalized_center_distance",
]
