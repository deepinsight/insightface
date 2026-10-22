from __future__ import annotations

from typing import Any

import numpy as np

from ..inference import FaceObservation


def clamp01(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 6)


def quality(face: FaceObservation) -> dict[str, float]:
    sharpness = clamp01(float(getattr(face, "sharpness", 0.0)))
    brightness = clamp01(float(getattr(face, "brightness", 0.0)))
    yaw = float(getattr(face, "yaw", 0.0))
    pitch = float(getattr(face, "pitch", 0.0))
    roll = float(getattr(face, "roll", 0.0))
    pose = clamp01(1.0 - min(1.0, (abs(yaw) + abs(pitch) + abs(roll)) / 135.0))
    provided = getattr(face, "quality_score", None)
    score = clamp01(
        float(provided)
        if provided is not None
        else 0.4 * sharpness + 0.35 * brightness + 0.25 * pose
    )
    return {
        "score": score,
        "sharpness": sharpness,
        "brightness": brightness,
        "pose": pose,
    }


def bounding_box(
    face: FaceObservation, image_width: int, image_height: int
) -> dict[str, Any]:
    left, top, right, bottom = face.bbox
    x = max(0, min(image_width, int(round(left))))
    y = max(0, min(image_height, int(round(top))))
    right_px = max(x, min(image_width, int(round(right))))
    bottom_px = max(y, min(image_height, int(round(bottom))))
    width = right_px - x
    height = bottom_px - y
    return {
        "pixels": {"x": x, "y": y, "width": width, "height": height},
        "normalized": {
            "left": round(x / image_width, 8),
            "top": round(y / image_height, 8),
            "width": round(width / image_width, 8),
            "height": round(height / image_height, 8),
        },
    }


def face_result(
    face: FaceObservation, image_width: int, image_height: int
) -> dict[str, Any]:
    landmarks = []
    if face.landmarks is not None:
        landmarks = [
            [round(float(point[0]), 3), round(float(point[1]), 3)]
            for point in face.landmarks
        ]
    return {
        "bbox": bounding_box(face, image_width, image_height),
        "landmarks": landmarks,
        "detection_score": clamp01(face.confidence),
        "quality": quality(face),
        **({"liveness": dict(face.liveness)} if face.liveness is not None else {}),
    }
