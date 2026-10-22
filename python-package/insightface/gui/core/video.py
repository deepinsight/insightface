"""Simple video helpers for local processing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


def timestamp_hhmmss(timestamp_ms: int) -> str:
    seconds = timestamp_ms // 1000
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def iter_video_frames(path: str | Path, frame_interval: int = 10) -> Iterator[Tuple[int, int, np.ndarray]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Unable to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % max(1, int(frame_interval)) == 0:
                timestamp_ms = int(index / fps * 1000)
                yield index, timestamp_ms, frame
            index += 1
    finally:
        cap.release()


def read_video_thumbnail(path: str | Path) -> np.ndarray | None:
    """Read the first decodable frame from a video for GUI preview."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
