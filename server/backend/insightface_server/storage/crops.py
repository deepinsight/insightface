from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np


class FaceCropStore:
    MEDIA_TYPE = "image/jpeg"

    def __init__(self, data_dir: Path, *, enabled: bool):
        self.data_dir = data_dir
        self.root = data_dir / "face-crops"
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.root.chmod(0o700)

    @staticmethod
    def _directory(collection_id: str) -> str:
        return hashlib.sha256(collection_id.encode()).hexdigest()

    @staticmethod
    def encode(
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> bytes:
        """Return a 112x112 JPEG crop without writing it to disk."""

        height, width = image.shape[:2]
        left = max(0, min(width, int(bbox[0])))
        top = max(0, min(height, int(bbox[1])))
        right = max(0, min(width, int(np.ceil(bbox[2]))))
        bottom = max(0, min(height, int(np.ceil(bbox[3]))))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise RuntimeError("Cannot persist an empty face crop")
        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Cannot encode face crop")
        return encoded.tobytes()

    def _managed_path(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        root = self.root.resolve()
        target = (self.data_dir / relative).resolve()
        if relative.is_absolute() or root not in target.parents:
            raise RuntimeError("Face crop path is outside managed storage")
        return target

    def save(
        self,
        collection_id: str,
        face_id: str,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> str | None:
        if not self.enabled:
            return None
        encoded = self.encode(image, bbox)
        directory = self.root / self._directory(collection_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = directory / f"{face_id}.jpg"
        temporary = directory / f".{face_id}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
            temporary.chmod(0o600)
            os.replace(temporary, target)
            target.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
        return str(target.relative_to(self.data_dir))

    def read(self, relative_path: str | None) -> bytes | None:
        """Read a legacy on-disk crop while rejecting paths outside the store."""

        if not relative_path:
            return None
        target = self._managed_path(relative_path)
        try:
            return target.read_bytes()
        except FileNotFoundError:
            return None

    def delete(self, relative_path: str | None) -> None:
        if not relative_path:
            return
        target = self._managed_path(relative_path)
        target.unlink(missing_ok=True)

    def delete_collection(self, collection_id: str) -> None:
        directory = self.root / self._directory(collection_id)
        if directory.is_dir():
            shutil.rmtree(directory)
