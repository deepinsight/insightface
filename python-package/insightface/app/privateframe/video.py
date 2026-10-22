"""Video metadata and output-path safety helpers."""

from __future__ import annotations

import os
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


def paths_refer_to_same_location(first: str | Path, second: str | Path) -> bool:
    left = Path(first).expanduser()
    right = Path(second).expanduser()
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            pass
    left_name = unicodedata.normalize("NFC", left.name).casefold()
    right_name = unicodedata.normalize("NFC", right.name).casefold()
    if left_name != right_name:
        return False
    try:
        if left.parent.exists() and right.parent.exists():
            return os.path.samefile(left.parent, right.parent)
    except OSError:
        pass
    try:
        left_parent = left.parent.resolve(strict=False)
        right_parent = right.parent.resolve(strict=False)
    except OSError:
        left_parent = left.parent.absolute()
        right_parent = right.parent.absolute()
    return unicodedata.normalize("NFC", str(left_parent)).casefold() == (
        unicodedata.normalize("NFC", str(right_parent)).casefold()
    )


def paths_are_distinct(paths: list[str | Path]) -> bool:
    return all(
        not paths_refer_to_same_location(paths[left], paths[right])
        for left in range(len(paths))
        for right in range(left + 1, len(paths))
    )


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_video(path: str | Path) -> VideoMetadata:
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0.0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return VideoMetadata(str(video_path), width, height, fps, frame_count, frame_count / fps)


def temporary_video_path(destination: str | Path) -> Path:
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=output.suffix, dir=output.parent)
    os.close(handle)
    path = Path(name)
    path.unlink()
    return path


__all__ = [
    "VideoMetadata",
    "paths_are_distinct",
    "paths_refer_to_same_location",
    "probe_video",
    "temporary_video_path",
]
