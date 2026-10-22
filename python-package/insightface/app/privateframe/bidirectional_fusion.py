"""Direction-neutral fusion for detector-bounded optical-flow gaps."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import iou, normalized_center_distance


def _interpolate(first: Any, second: Any, fraction: float) -> np.ndarray:
    return (
        np.asarray(first, dtype=np.float64) * (1.0 - fraction)
        + np.asarray(second, dtype=np.float64) * fraction
    )


def interpolate_published_geometry(
    center_box: Any,
    left_endpoint_box: Any,
    right_endpoint_box: Any,
    fraction: float,
) -> np.ndarray:
    """Combine a motion-derived center with detector-confirmed endpoint size.

    The center remains entirely controlled by the direction-neutral flow
    fusion.  Width and height are interpolated geometrically (linearly in log
    space), which is invariant under endpoint exchange plus ``t -> 1-t`` and
    cannot overshoot either positive endpoint size.

    Whether endpoint geometry is trustworthy is deliberately *not* decided
    here.  The streaming policy calls this helper only after its edge-shrink
    bridge has verified the endpoint geometry and bidirectional gap evidence.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("bidirectional fraction must be in [0, 1]")
    center = np.asarray(center_box, dtype=np.float64)
    left = np.asarray(left_endpoint_box, dtype=np.float64)
    right = np.asarray(right_endpoint_box, dtype=np.float64)
    if center.shape != (4,) or left.shape != (4,) or right.shape != (4,):
        raise ValueError("bidirectional geometry boxes must have four values")

    left_size = left[2:4] - left[0:2]
    right_size = right[2:4] - right[0:2]
    if np.any(left_size <= 0.0) or np.any(right_size <= 0.0):
        raise ValueError("endpoint geometry boxes must have positive size")

    size = np.exp(np.log(left_size) * (1.0 - fraction) + np.log(right_size) * fraction)
    center_xy = 0.5 * (center[0:2] + center[2:4])
    return np.concatenate((center_xy - 0.5 * size, center_xy + 0.5 * size))


def _review_values(
    local_review: dict[str, Any] | None,
) -> tuple[Any | None, bool, float]:
    if local_review is None:
        return None, False, 0.0
    review_box = local_review.get("box", local_review.get("local_box"))
    matched = bool(local_review.get("matched", review_box is not None) and review_box is not None)
    confidence = float(local_review.get("confidence", local_review.get("local_confidence") or 0.0))
    return review_box, matched, confidence


def _smoothstep(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("soft_confidence_high must exceed soft_confidence_low")
    normalized = min(1.0, max(0.0, (value - lower) / (upper - lower)))
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _centered_triangular(values: np.ndarray, radius: int) -> np.ndarray:
    """Apply a finite, time-reversal-symmetric triangular filter."""

    if radius < 0:
        raise ValueError("soft_bias_radius must be non-negative")
    if radius == 0 or values.size == 0:
        return values.copy()
    ascending = np.arange(1, radius + 2, dtype=np.float64)
    kernel = np.concatenate((ascending, ascending[-2::-1]))
    kernel /= float(kernel.sum())
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def soft_fuse_bidirectional_sequence(
    records: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Softly fuse a detector-bounded gap without privileging a direction.

    Each input record contains ``frame_idx``, the normalized endpoint
    ``fraction``, ``forward`` and ``reverse`` candidates with ``box`` fields,
    and an optional ``local_review``.  Local SCRFD evidence contributes only
    when the two directions disagree and the review decisively supports one
    of them.  A centered triangular filter spreads that signed support without
    introducing a causal direction.

    If all signed support is zero, the output is exactly the temporal
    interpolation. Reversing record order, replacing ``fraction`` with
    ``1-fraction`` and swapping forward/reverse produces identical boxes in
    reverse order.
    """

    if not records:
        return []
    beta = float(settings["soft_bias_beta"])
    radius = int(settings["soft_bias_radius"])
    confidence_low = float(settings["soft_confidence_low"])
    confidence_high = float(settings["soft_confidence_high"])
    # Validate these even when every local review is absent so a malformed
    # configuration fails deterministically.
    _smoothstep(confidence_low, confidence_low, confidence_high)
    if beta < 0.0:
        raise ValueError("soft_bias_beta must be non-negative")
    if radius < 0:
        raise ValueError("soft_bias_radius must be non-negative")

    mutual_iou = float(settings["mutual_min_iou"])
    mutual_distance = float(settings["mutual_max_center_distance"])
    local_confidence = float(settings["local_review_min_confidence"])
    local_iou = float(settings["local_review_min_iou"])
    local_margin = float(settings["local_review_min_iou_margin"])

    fractions: list[float] = []
    forward_boxes: list[np.ndarray] = []
    reverse_boxes: list[np.ndarray] = []
    raw_biases: list[float] = []
    previous_fraction = -math.inf
    for record in records:
        fraction = float(record["fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("bidirectional fraction must be in [0, 1]")
        if fraction < previous_fraction:
            raise ValueError("bidirectional sequence fractions must be non-decreasing")
        previous_fraction = fraction
        forward_box = np.asarray(record["forward"]["box"], dtype=np.float64)
        reverse_box = np.asarray(record["reverse"]["box"], dtype=np.float64)
        if forward_box.shape != (4,) or reverse_box.shape != (4,):
            raise ValueError("bidirectional candidate boxes must have four values")
        fractions.append(fraction)
        forward_boxes.append(forward_box)
        reverse_boxes.append(reverse_box)

        overlap = iou(forward_box, reverse_box)
        distance = normalized_center_distance(forward_box, reverse_box)
        review_box, matched, confidence = _review_values(record.get("local_review"))
        signed_support = 0.0
        if (
            overlap < mutual_iou
            and distance > mutual_distance
            and matched
            and review_box is not None
            and confidence >= local_confidence
        ):
            forward_overlap = iou(review_box, forward_box)
            reverse_overlap = iou(review_box, reverse_box)
            difference = reverse_overlap - forward_overlap
            if max(forward_overlap, reverse_overlap) >= local_iou and abs(difference) >= local_margin:
                confidence_gain = _smoothstep(confidence, confidence_low, confidence_high)
                signed_support = confidence_gain * difference
        raw_biases.append(signed_support)

    smoothed_biases = _centered_triangular(np.asarray(raw_biases, dtype=np.float64), radius)
    outputs: list[dict[str, Any]] = []
    for record, fraction, forward_box, reverse_box, raw_bias, bias in zip(
        records,
        fractions,
        forward_boxes,
        reverse_boxes,
        raw_biases,
        smoothed_biases,
        strict=True,
    ):
        if bias == 0.0 or fraction in {0.0, 1.0}:
            # Preserve neutral temporal interpolation when local evidence contributes
            # nothing, and keep detector endpoints exact.
            reverse_weight = fraction
        else:
            prior_logit = math.log(fraction / (1.0 - fraction))
            reverse_weight = _sigmoid(prior_logit + beta * float(bias))
        value = _interpolate(forward_box, reverse_box, reverse_weight)
        outputs.append(
            {
                "frame_idx": int(record["frame_idx"]),
                "fraction": fraction,
                "box": value.tolist(),
                "weight": float(reverse_weight),
                "forward_weight": float(1.0 - reverse_weight),
                "reverse_weight": float(reverse_weight),
                "raw_bias": float(raw_bias),
                "bias": float(bias),
            }
        )
    return outputs


__all__ = [
    "interpolate_published_geometry",
    "soft_fuse_bidirectional_sequence",
]
