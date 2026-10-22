"""Camera availability helpers."""

from __future__ import annotations

from typing import List


def list_camera_indices(max_indices: int = 4) -> List[int]:
    try:
        import cv2
    except Exception:
        return []
    indices = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            indices.append(idx)
        cap.release()
    return indices
