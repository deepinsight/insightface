from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
from io import BytesIO

import cv2
import numpy as np
from anyio import to_thread
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import Settings
from ..errors import ApiError


@dataclass(frozen=True, slots=True)
class ImageData:
    content: bytes
    sha256: str
    pixels: np.ndarray
    filename: str


class ImageLoader:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def from_upload(self, upload: UploadFile) -> ImageData:
        chunks = bytearray()
        while chunk := await upload.read(1024 * 1024):
            if len(chunks) + len(chunk) > self.settings.max_image_bytes:
                raise ApiError(
                    "image_too_large",
                    "The image exceeds the configured byte limit.",
                    413,
                    {"filename": upload.filename},
                )
            chunks.extend(chunk)
        return await to_thread.run_sync(
            partial(self.from_bytes, bytes(chunks), filename=upload.filename or "upload"),
            abandon_on_cancel=True,
        )

    def from_bytes(self, content: bytes, *, filename: str = "image") -> ImageData:
        if not content:
            raise ApiError("invalid_image", "The image is empty.", 422)
        if len(content) > self.settings.max_image_bytes:
            raise ApiError("image_too_large", "The image exceeds the byte limit.", 413)
        try:
            with Image.open(BytesIO(content)) as source:
                width, height = source.size
                image_format = source.format
                if width <= 0 or height <= 0 or width * height > self.settings.max_image_pixels:
                    raise ApiError("image_too_large", "The image pixel limit was exceeded.", 413)
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                pixels = cv2.cvtColor(np.asarray(oriented), cv2.COLOR_RGB2BGR)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ApiError("invalid_image", "The image could not be decoded.", 422) from exc
        if image_format not in {"JPEG", "PNG", "WEBP", "BMP"}:
            raise ApiError(
                "invalid_image",
                "Only JPEG, PNG, WebP, and BMP images are supported.",
                422,
            )
        if pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ApiError("invalid_image", "The image could not be decoded.", 422)
        if int(pixels.shape[0]) * int(pixels.shape[1]) > self.settings.max_image_pixels:
            raise ApiError("image_too_large", "The image pixel limit was exceeded.", 413)
        return ImageData(
            content=content,
            sha256=hashlib.sha256(content).hexdigest(),
            pixels=pixels,
            filename=filename,
        )
