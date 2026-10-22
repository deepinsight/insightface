from __future__ import annotations

import hashlib
from collections.abc import Callable
from io import BytesIO

import numpy as np
import pytest
from insightface_server.config import Settings
from insightface_server.errors import ApiError
from insightface_server.services.images import ImageLoader
from PIL import Image


def encoded_image(image: Image.Image, image_format: str = "BMP") -> bytes:
    stream = BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


@pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA"])
def test_bmp_normalizes_pixels_to_uint8_bgr(
    make_settings: Callable[..., Settings], mode: str
) -> None:
    # Non-square dimensions exercise BMP row padding; distinct colors check BGR
    # channel order and row orientation, including indexed-color BMP palettes.
    pixels = np.asarray(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[20, 40, 60], [200, 150, 100], [0, 0, 0]]],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels).convert(mode)
    content = encoded_image(image)
    loaded = ImageLoader(make_settings()).from_bytes(content, filename="claimed.png")

    expected = np.asarray(image.convert("RGB"))[:, :, ::-1]
    np.testing.assert_array_equal(loaded.pixels, expected)
    assert loaded.pixels.dtype == np.uint8
    assert loaded.pixels.flags.c_contiguous
    assert loaded.content == content
    assert loaded.sha256 == hashlib.sha256(content).hexdigest()
    assert loaded.filename == "claimed.png"


@pytest.mark.parametrize("limit", ["max_image_bytes", "max_image_pixels"])
def test_bmp_preserves_byte_and_pixel_limits(
    make_settings: Callable[..., Settings], limit: str
) -> None:
    content = encoded_image(Image.new("RGB", (9, 7), "red"))
    boundary = len(content) if limit == "max_image_bytes" else 9 * 7
    accepted = ImageLoader(make_settings(**{limit: boundary})).from_bytes(content)
    assert accepted.pixels.shape == (7, 9, 3)

    with pytest.raises(ApiError) as rejected:
        ImageLoader(make_settings(**{limit: boundary - 1})).from_bytes(content)

    assert rejected.value.code == "image_too_large"
    assert rejected.value.status_code == 413


@pytest.mark.parametrize("image_format", ["GIF", "TIFF"])
def test_bmp_filename_does_not_allow_other_formats(
    make_settings: Callable[..., Settings], image_format: str
) -> None:
    content = encoded_image(Image.new("RGB", (9, 7), "red"), image_format)

    with pytest.raises(ApiError) as rejected:
        ImageLoader(make_settings()).from_bytes(content, filename="claimed.bmp")

    assert rejected.value.code == "invalid_image"
    assert rejected.value.status_code == 422
    assert "BMP" in rejected.value.message


@pytest.mark.parametrize("content", [b"not an image", b"BM\x00\x00", b""])
def test_invalid_bmp_is_rejected(make_settings: Callable[..., Settings], content: bytes) -> None:
    with pytest.raises(ApiError) as rejected:
        ImageLoader(make_settings()).from_bytes(content, filename="invalid.bmp")

    assert rejected.value.code == "invalid_image"
    assert rejected.value.status_code == 422


def test_truncated_bmp_pixels_are_rejected(make_settings: Callable[..., Settings]) -> None:
    content = encoded_image(Image.new("RGB", (9, 7), "red"))

    with pytest.raises(ApiError) as rejected:
        ImageLoader(make_settings()).from_bytes(content[:-20], filename="truncated.bmp")

    assert rejected.value.code == "invalid_image"
    assert rejected.value.status_code == 422
