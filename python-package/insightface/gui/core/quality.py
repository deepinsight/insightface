"""Heuristic face quality scoring."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.shape[2] >= 3:
        return (0.114 * arr[..., 0] + 0.587 * arr[..., 1] + 0.299 * arr[..., 2]).astype(np.float32)
    return arr[..., 0].astype(np.float32)


def blur_score(image: np.ndarray) -> float:
    gray = _to_gray(image)
    try:
        import cv2

        value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        value = float(np.var(gx) + np.var(gy))
    return max(0.0, value)


def _bbox_area_ratio(bbox: Iterable[float], image_shape: Tuple[int, ...]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    face_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = float(max(1, image_shape[0] * image_shape[1]))
    return face_area / image_area


def _frontal_score(kps: Optional[Iterable[Iterable[float]]]) -> tuple[float, List[str]]:
    if kps is None:
        return 0.55, ["landmarks_missing"]
    pts = np.asarray(list(kps), dtype=np.float32)
    if pts.shape[0] < 5:
        return 0.55, ["landmarks_incomplete"]
    left_eye, right_eye, nose, left_mouth, right_mouth = pts[:5]
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0
    eye_distance = max(float(np.linalg.norm(left_eye - right_eye)), 1.0)
    nose_offset = abs(float(nose[0] - eye_mid[0])) / eye_distance
    vertical_span = max(abs(float(mouth_mid[1] - eye_mid[1])), 1.0)
    nose_vertical = abs(float(nose[1] - (eye_mid[1] + mouth_mid[1]) / 2.0)) / vertical_span
    score = 1.0 - min(0.7, nose_offset) - min(0.3, nose_vertical * 0.3)
    flags: List[str] = []
    if nose_offset > 0.35:
        flags.append("side_face")
    return max(0.0, min(1.0, score)), flags


def score_face(
    image: np.ndarray,
    bbox: Iterable[float],
    kps: Optional[Iterable[Iterable[float]]] = None,
    det_score: float = 1.0,
) -> tuple[float, List[str]]:
    flags: List[str] = []
    det_component = max(0.0, min(1.0, float(det_score)))
    if det_component < 0.55:
        flags.append("low_detection_score")

    area_ratio = _bbox_area_ratio(bbox, image.shape)
    if area_ratio < 0.01:
        flags.append("face_too_small")
    size_component = max(0.0, min(1.0, area_ratio / 0.08))

    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else image

    blur = blur_score(crop)
    if blur < 45:
        flags.append("blurry")
    blur_component = max(0.0, min(1.0, blur / 250.0))

    gray = _to_gray(crop)
    brightness = float(np.mean(gray)) if gray.size else 128.0
    exposure_component = 1.0 - min(1.0, abs(brightness - 128.0) / 128.0)
    if brightness < 45:
        flags.append("too_dark")
    elif brightness > 220:
        flags.append("too_bright")

    frontal_component, frontal_flags = _frontal_score(kps)
    flags.extend(frontal_flags)

    score = (
        0.28 * det_component
        + 0.22 * size_component
        + 0.22 * blur_component
        + 0.18 * frontal_component
        + 0.10 * exposure_component
    )
    return max(0.0, min(1.0, float(score))), sorted(set(flags))
