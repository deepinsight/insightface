"""General utilities for image and file handling."""

from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_image(path: str | os.PathLike[str]) -> Optional[np.ndarray]:
    file_path = Path(path)
    try:
        import cv2

        data = np.fromfile(str(file_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        try:
            from PIL import Image

            with Image.open(file_path) as image:
                rgb = image.convert("RGB")
                arr = np.asarray(rgb)
                return arr[..., ::-1].copy()
        except Exception:
            return None


def save_image(path: str | os.PathLike[str], image: np.ndarray) -> bool:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        ext = file_path.suffix or ".png"
        ok, data = cv2.imencode(ext, image)
        if not ok:
            return False
        data.tofile(str(file_path))
        return True
    except Exception:
        try:
            from PIL import Image

            rgb = image[..., ::-1] if image.ndim == 3 and image.shape[2] >= 3 else image
            Image.fromarray(rgb).save(file_path)
            return True
        except Exception:
            return False


def encode_webp_thumbnail(
    image: np.ndarray | None,
    max_side: int,
    quality: int,
) -> bytes | None:
    """Encode an image array as a small WebP thumbnail.

    GUI image arrays are BGR when loaded through OpenCV. The helper converts to
    RGB for Pillow, keeps aspect ratio, and returns compressed bytes for SQLite.
    """

    if image is None or image.size == 0:
        return None
    try:
        from PIL import Image

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3][..., ::-1]
        pil_image = Image.fromarray(arr).convert("RGB")
        width, height = pil_image.size
        longest = max(width, height)
        if longest <= 0:
            return None
        scale = min(1.0, float(max_side) / float(longest))
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        if target != pil_image.size:
            pil_image = pil_image.resize(target, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        pil_image.save(buffer, format="WEBP", quality=int(quality), method=6)
        return buffer.getvalue()
    except Exception:
        return None


def crop_bbox(image: np.ndarray, bbox: Iterable[float], padding: float = 0.10) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = x2 - x1
    height = y2 - y1
    x1 -= width * padding
    x2 += width * padding
    y1 -= height * padding
    y2 += height * padding
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(image.shape[1], int(round(x2))), min(image.shape[0], int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return image.copy()
    return image[iy1:iy2, ix1:ix2].copy()


def file_sha256(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json_dumps(data: Any) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    return json.dumps(data, ensure_ascii=False, default=default)


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_images(folder: str | os.PathLike[str], recursive: bool = True) -> list[Path]:
    from .constants import IMAGE_EXTENSIONS

    root = Path(folder)
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def human_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"
