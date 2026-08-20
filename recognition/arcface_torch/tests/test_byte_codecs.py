"""backbones/byte_codecs.py: GPU-side TIFF / PNG byte-stream writers that reproduce PIL byte-for-byte
(so TransFace++ can train on file bytes without a per-image CPU encode)."""
import io
import sys
sys.path.insert(0, ".")
import numpy as np
import pytest
import torch
from PIL import Image


def _pil(a, fmt):
    b = io.BytesIO()
    Image.fromarray(a).save(b, **({"format": "PNG", "compress_level": 0} if fmt == "png" else {"format": "TIFF"}))
    return np.frombuffer(b.getvalue(), np.uint8)


def _images():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
    grad = np.tile(np.linspace(0, 255, 112).astype(np.uint8)[None, :, None], (112, 1, 3))
    flat = np.full((112, 112, 3), 77, np.uint8)
    smooth = (rng.integers(0, 256, (14, 14, 3)).repeat(8, 0).repeat(8, 1)).astype(np.uint8)
    return [noise, grad, flat, smooth]


@pytest.mark.parametrize("fmt", ["tiff", "png"])
def test_codec_matches_pil_byte_for_byte(fmt):
    from backbones.byte_codecs import encode_image_bytes
    imgs = _images()
    out = encode_image_bytes(torch.from_numpy(np.stack(imgs)), fmt)        # (B, H, W, C) uint8 -> (B, L) int64
    assert out.dtype == torch.long
    for i, a in enumerate(imgs):
        ref = _pil(a, fmt)
        assert out.shape[1] == ref.shape[0], (fmt, out.shape, ref.shape)
        mism = np.nonzero(out[i].numpy() != ref)[0]
        assert mism.size == 0, f"{fmt} image {i}: {mism.size} mismatching bytes, first at {mism[:5]}"


def test_png_filter_choice_matches_pil_per_row():
    from backbones.byte_codecs import png_filter_scanlines
    import zlib
    for a in _images():
        ref = _pil(a, "png")
        idat = ref[41:41 + int.from_bytes(ref[33:37].tobytes(), "big")].tobytes()
        rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(112, 1 + 336)
        got = png_filter_scanlines(torch.from_numpy(a)[None])[0].numpy().reshape(112, 337)
        assert np.array_equal(got[:, 0], rows[:, 0]), np.nonzero(got[:, 0] != rows[:, 0])[0][:5]
        assert np.array_equal(got, rows)


def test_image_to_bytes_uses_codecs_and_keeps_device():
    from backbones.transface_pp import image_to_bytes
    a = torch.from_numpy(np.stack(_images()[:2]))                            # (B, H, W, C) uint8
    img = ((a.permute(0, 3, 1, 2).float() / 255) - 0.5) / 0.5
    for fmt in ("tiff", "png"):
        out = image_to_bytes(img, fmt)
        assert out.device == img.device
        for i in range(2):
            assert np.array_equal(out[i].numpy(), _pil(a[i].numpy(), fmt)), fmt


def test_adler32_and_crc32_match_zlib():
    import zlib
    from backbones.byte_codecs import adler32, crc32
    rng = np.random.default_rng(1)
    data = torch.from_numpy(rng.integers(0, 256, (3, 5000), dtype=np.uint8))
    for i in range(3):
        assert int(adler32(data)[i]) == zlib.adler32(data[i].numpy().tobytes())
        assert int(crc32(data)[i]) == zlib.crc32(data[i].numpy().tobytes())
