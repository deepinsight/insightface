from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .base import FaceObservation


@dataclass(frozen=True, slots=True)
class RegistrationQualityPolicy:
    min_face_size: int = 40
    min_detection_score: float = 0.50
    min_quality_score: float = 0.25
    min_sharpness: float = 0.08
    min_brightness: float = 0.06
    max_brightness: float = 0.94
    max_abs_pose_degrees: float = 45.0


def enrich_quality(image: np.ndarray, face: FaceObservation) -> None:
    """Populate deterministic, documented quality heuristics on a face result."""

    image_height, image_width = image.shape[:2]
    left, top, right, bottom = face.bbox
    x1, y1 = max(0, int(math.floor(left))), max(0, int(math.floor(top)))
    x2 = min(image_width, int(math.ceil(right)))
    y2 = min(image_height, int(math.ceil(bottom)))
    crop = image[y1:y2, x1:x2]
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        face.brightness = float(np.clip(np.mean(gray) / 255.0, 0.0, 1.0))
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        face.sharpness = float(np.clip(1.0 - math.exp(-laplacian_variance / 100.0), 0, 1))

    if face.landmarks is not None and face.landmarks.shape == (5, 2):
        points = face.landmarks.astype(np.float64)
        eyes_midpoint = (points[0] + points[1]) / 2.0
        mouth_midpoint = (points[3] + points[4]) / 2.0
        eye_delta = points[1] - points[0]
        eye_distance = max(float(np.linalg.norm(eye_delta)), 1e-6)
        face.roll = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))
        face.yaw = float((points[2, 0] - eyes_midpoint[0]) / eye_distance * 90.0)
        vertical = max(float(mouth_midpoint[1] - eyes_midpoint[1]), 1e-6)
        face.pitch = float(
            ((points[2, 1] - eyes_midpoint[1]) / vertical - 0.45) * 90.0
        )

    face_width = max(0.0, right - left)
    face_height = max(0.0, bottom - top)
    size_score = float(np.clip(min(face_width, face_height) / 80.0, 0.0, 1.0))
    brightness_score = float(np.clip(1.0 - abs(face.brightness - 0.5) / 0.5, 0.0, 1.0))
    pose_score = float(
        np.clip(
            1.0 - max(abs(face.yaw), abs(face.pitch), abs(face.roll)) / 90.0,
            0.0,
            1.0,
        )
    )
    face.quality_score = float(
        np.clip(
            0.35 * face.detection_score
            + 0.25 * face.sharpness
            + 0.15 * brightness_score
            + 0.15 * size_score
            + 0.10 * pose_score,
            0.0,
            1.0,
        )
    )


def registration_rejection_reasons(
    face: FaceObservation,
    policy: RegistrationQualityPolicy | None = None,
) -> list[str]:
    """Return stable public reason codes for a candidate registration face."""

    selected = policy or RegistrationQualityPolicy()
    width = max(0.0, face.bbox[2] - face.bbox[0])
    height = max(0.0, face.bbox[3] - face.bbox[1])
    reasons: list[str] = []
    if min(width, height) < selected.min_face_size:
        reasons.append("face_too_small")
    if face.detection_score < selected.min_detection_score:
        reasons.append("low_detection_score")
    if face.sharpness < selected.min_sharpness:
        reasons.append("low_quality")
    if not selected.min_brightness <= face.brightness <= selected.max_brightness:
        if "low_quality" not in reasons:
            reasons.append("low_quality")
    if face.quality_score < selected.min_quality_score and "low_quality" not in reasons:
        reasons.append("low_quality")
    if max(abs(face.yaw), abs(face.pitch), abs(face.roll)) > selected.max_abs_pose_degrees:
        reasons.append("extreme_pose")
    return reasons
