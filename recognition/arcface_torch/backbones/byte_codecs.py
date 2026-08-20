"""Batched, device-side writers for the image file formats TransFace++ trains on.

`encode_image_bytes(images, fmt)` turns a (B, H, W, C) uint8 batch into the exact
byte streams PIL would write - TIFF (uncompressed) and PNG (compress_level=0,
i.e. stored deflate blocks with PIL's adaptive per-row filtering) - without a
per-image CPU encode. The container layout only depends on the image size, so it
is probed once with PIL and cached; pixel-dependent parts (scanline filtering,
Adler-32) are computed with tensor ops on the images' device. CRC-32 of the PNG
IDAT chunk is computed with zlib on the CPU (a few microseconds per image).
"""
import io
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F

_LAYOUT_CACHE = {}


def _pil_probe(h, w, c, fmt):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, c), np.uint8)).save(
        buf, **({"format": "PNG", "compress_level": 0} if fmt == "png" else {"format": "TIFF"}))
    return np.frombuffer(buf.getvalue(), np.uint8)


# --------------------------------------------------------------------------- #
# checksums
# --------------------------------------------------------------------------- #
def adler32(data: torch.Tensor) -> torch.Tensor:
    """Adler-32 of each row of a (B, L) uint8 tensor, vectorised (closed form)."""
    d = data.long()
    n = d.shape[1]
    a = (1 + d.sum(dim=1)) % 65521
    weights = torch.arange(n, 0, -1, device=d.device, dtype=torch.long)   # n, n-1, ..., 1
    b = (n + (d * weights).sum(dim=1)) % 65521
    return (b << 16) | a


def crc32(data: torch.Tensor) -> torch.Tensor:
    """CRC-32 (zlib) of each row of a (B, L) uint8 tensor. Computed on the CPU with zlib."""
    rows = data.to(torch.uint8).cpu().numpy()
    out = torch.tensor([zlib.crc32(r.tobytes()) & 0xFFFFFFFF for r in rows], dtype=torch.long)
    return out.to(data.device)


def _be32(x: torch.Tensor) -> torch.Tensor:
    """(B,) int -> (B, 4) big-endian bytes."""
    return torch.stack([(x >> 24) & 255, (x >> 16) & 255, (x >> 8) & 255, x & 255], dim=1)


# --------------------------------------------------------------------------- #
# TIFF
# --------------------------------------------------------------------------- #
def _tiff_header(h, w, c, device):
    key = ("tiff", h, w, c)
    if key not in _LAYOUT_CACHE:
        probe = _pil_probe(h, w, c, "tiff")
        n_pix = h * w * c
        assert probe.shape[0] >= n_pix and np.all(probe[-n_pix:] == 0)
        _LAYOUT_CACHE[key] = torch.from_numpy(probe[:-n_pix].astype(np.int64))
    return _LAYOUT_CACHE[key].to(device)


def encode_tiff_bytes(images: torch.Tensor) -> torch.Tensor:
    """(B, H, W, C) uint8 -> (B, L) int64 identical to PIL's uncompressed TIFF (header + raw HWC bytes)."""
    B, h, w, c = images.shape
    header = _tiff_header(h, w, c, images.device).expand(B, -1)
    return torch.cat([header, images.reshape(B, -1).long()], dim=1)


# --------------------------------------------------------------------------- #
# PNG (compress_level = 0)
# --------------------------------------------------------------------------- #
def png_filter_scanlines(images: torch.Tensor) -> torch.Tensor:
    """PIL/ZipEncode adaptive filtering: per row try None, Up, Sub, Paeth (Average only
    with `optimize`), score = sum of min(v, 256 - v), keep the first minimum.
    (B, H, W, C) uint8 -> (B, H * (1 + W*C)) uint8 filtered scanlines."""
    B, h, w, c = images.shape
    bpp = c
    cur = images.reshape(B, h, w * c).to(torch.int16)
    up = F.pad(cur, (0, 0, 1, 0))[:, :-1]                                   # previous row (zeros for row 0)
    left = F.pad(cur, (bpp, 0))[:, :, :-bpp]
    ul = F.pad(up, (bpp, 0))[:, :, :-bpp]
    none = cur
    upf = (cur - up) & 255
    sub = (cur - left) & 255
    p = left + up - ul
    pa, pb, pc = (p - left).abs(), (p - up).abs(), (p - ul).abs()
    pred = torch.where((pa <= pb) & (pa <= pc), left, torch.where(pb <= pc, up, ul))
    paeth = (cur - pred) & 255
    cands = torch.stack([none, upf, sub, paeth], dim=2)                    # (B, H, 4, W*C)
    dist = torch.where(cands < 128, cands, 256 - cands).long().sum(dim=3)  # (B, H, 4)
    choice = dist.argmin(dim=2)                                             # first minimum
    ftype = torch.tensor([0, 2, 1, 4], device=images.device)[choice]        # PNG filter type ids
    rows = cands.gather(2, choice[:, :, None, None].expand(B, h, 1, w * c)).squeeze(2)
    out = torch.cat([ftype[:, :, None], rows], dim=2).to(torch.uint8)
    return out.reshape(B, -1)


def _png_layout(h, w, c):
    key = ("png", h, w, c)
    if key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[key]
    probe = _pil_probe(h, w, c, "png")
    pos, idat = 8, None
    while pos < probe.shape[0]:
        ln = struct.unpack(">I", probe[pos:pos + 4].tobytes())[0]
        typ = probe[pos + 4:pos + 8].tobytes()
        if typ == b"IDAT":
            assert idat is None, "multi-IDAT PNGs are not supported"
            idat = (pos, ln)
        pos += 12 + ln
    idat_pos, idat_len = idat
    data_start = idat_pos + 8
    z = probe[data_start:data_start + idat_len].tobytes()
    blocks, i, payload_off = [], 2, 0                                       # skip the 2-byte zlib header
    while True:
        bfinal, blen = z[i] & 1, struct.unpack("<H", z[i + 1:i + 3])[0]
        blocks.append((data_start + i + 5, payload_off, blen))              # (file offset, payload offset, length)
        payload_off += blen
        i += 5 + blen
        if bfinal:
            break
    layout = dict(
        template=torch.from_numpy(probe.astype(np.int64)),
        blocks=blocks,
        adler_pos=data_start + idat_len - 4,
        crc_pos=idat_pos + 8 + idat_len,
        crc_start=idat_pos + 4,                                             # CRC covers type + data
        payload_len=payload_off,
    )
    assert payload_off == h * (1 + w * c), (payload_off, h, w, c)
    _LAYOUT_CACHE[key] = layout
    return layout


def encode_png_bytes(images: torch.Tensor) -> torch.Tensor:
    """(B, H, W, C) uint8 -> (B, L) int64 identical to PIL's PNG with compress_level=0."""
    B, h, w, c = images.shape
    lay = _png_layout(h, w, c)
    dev = images.device
    out = lay["template"].to(dev).expand(B, -1).clone()
    payload = png_filter_scanlines(images)                                  # (B, payload_len) uint8
    for file_off, pay_off, ln in lay["blocks"]:
        out[:, file_off:file_off + ln] = payload[:, pay_off:pay_off + ln].long()
    out[:, lay["adler_pos"]:lay["adler_pos"] + 4] = _be32(adler32(payload))
    crc = crc32(out[:, lay["crc_start"]:lay["crc_pos"]].to(torch.uint8))
    out[:, lay["crc_pos"]:lay["crc_pos"] + 4] = _be32(crc)
    return out


def encode_image_bytes(images: torch.Tensor, fmt: str) -> torch.Tensor:
    """Dispatch: images (B, H, W, C) uint8 on any device -> (B, L) int64 file bytes on the same device."""
    if fmt == "tiff":
        return encode_tiff_bytes(images)
    if fmt == "png":
        return encode_png_bytes(images)
    raise ValueError(fmt)
