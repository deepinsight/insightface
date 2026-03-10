import os
import cv2
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple, Optional

# ----------------------------
# Profiles (tweak as you like)
# ----------------------------
@dataclass
class CCTVProfile:
    name: str
    # resolution loss
    scale_range: Tuple[float, float] = (0.25, 0.45)

    # tone/color
    gamma_range: Tuple[float, float] = (1.2, 1.8)
    sat_mul_range: Tuple[float, float] = (0.6, 0.85)
    val_mul_range: Tuple[float, float] = (0.85, 1.05)
    hue_shift_range: Tuple[float, float] = (-3.0, 3.0)     # degrees in HSV space (0..179), keep small
    color_cast_range: Tuple[float, float] = (-8.0, 8.0)    # added in BGR (per channel)

    # blur
    gaussian_sigma_range: Tuple[float, float] = (0.6, 1.4)
    motion_blur_prob: float = 0.35
    motion_blur_kernels: Tuple[int, ...] = (3, 5, 7, 9)

    # noise + compression
    noise_sigma_range: Tuple[float, float] = (3.0, 12.0)
    jpeg_quality_range: Tuple[int, int] = (15, 45)

    # optional: chroma smear (video-ish)
    chroma_smear_prob: float = 0.3
    chroma_smear_sigma_range: Tuple[float, float] = (0.8, 2.0)

    # optional: vignetting
    vignette_prob: float = 0.2
    vignette_strength_range: Tuple[float, float] = (0.15, 0.35)

    # optional: slight resize jitter (simulates different crop sizes)
    resize_jitter_prob: float = 0.25
    resize_jitter_range: Tuple[float, float] = (0.90, 1.05)


DAYLIGHT = CCTVProfile(
    name="daylight",
    scale_range=(0.30, 0.55),
    gamma_range=(1.0, 1.4),
    sat_mul_range=(0.65, 0.95),
    noise_sigma_range=(2.0, 8.0),
    jpeg_quality_range=(20, 55),
    motion_blur_prob=0.25,
)

NIGHT = CCTVProfile(
    name="night",
    scale_range=(0.20, 0.45),
    gamma_range=(1.3, 2.2),
    sat_mul_range=(0.45, 0.80),
    val_mul_range=(0.70, 1.00),
    noise_sigma_range=(6.0, 18.0),
    jpeg_quality_range=(10, 40),
    motion_blur_prob=0.45,
    vignette_prob=0.35,
)

IR = CCTVProfile(
    name="ir",
    scale_range=(0.20, 0.45),
    gamma_range=(1.2, 2.0),
    sat_mul_range=(0.05, 0.25),      # very low color
    hue_shift_range=(-1.0, 1.0),
    noise_sigma_range=(4.0, 14.0),
    jpeg_quality_range=(10, 40),
    chroma_smear_prob=0.15,
    motion_blur_prob=0.30,
)


# ----------------------------
# Core augmentation utilities
# ----------------------------
def _apply_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)

def _motion_blur(img: np.ndarray, k: int, angle_rad: float) -> np.ndarray:
    kernel = np.zeros((k, k), dtype=np.float32)
    cx = cy = k // 2
    for i in range(k):
        x = int(cx + (i - cx) * np.cos(angle_rad))
        y = int(cy + (i - cy) * np.sin(angle_rad))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)

def _jpeg_recompress(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def _chroma_smear(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    # Blur chroma channels more than luma: simulate 4:2:0 + ISP chroma filtering
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycrcb[..., 0]
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]
    k = int(2 * round(3 * sigma) + 1)
    cr = cv2.GaussianBlur(cr, (k, k), sigmaX=sigma)
    cb = cv2.GaussianBlur(cb, (k, k), sigmaX=sigma)
    ycrcb[..., 1] = cr
    ycrcb[..., 2] = cb
    out = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out

def _vignette(img: np.ndarray, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    dist /= dist.max()
    mask = (1.0 - strength * (dist ** 1.5)).astype(np.float32)
    out = img.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def _resize_jitter(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    nw, nh = int(w * factor), int(h * factor)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # center crop/pad back to original size
    out = np.zeros_like(img)
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    crop = resized[y0:y0 + h, x0:x0 + w]
    if crop.shape[0] == h and crop.shape[1] == w:
        return crop
    # padding case
    y1 = (h - crop.shape[0]) // 2
    x1 = (w - crop.shape[1]) // 2
    out[y1:y1 + crop.shape[0], x1:x1 + crop.shape[1]] = crop
    return out


def cctv_augment(img_bgr: np.ndarray, profile: CCTVProfile, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()

    # (optional) slight size jitter (keeps identity mostly)
    if random.random() < profile.resize_jitter_prob:
        factor = random.uniform(*profile.resize_jitter_range)
        img = _resize_jitter(img, factor)

    # 1) resolution loss: downscale -> upscale
    scale = random.uniform(*profile.scale_range)
    small = cv2.resize(img, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2) tone: gamma (darken)
    gamma = random.uniform(*profile.gamma_range)
    img = _apply_gamma(img, gamma)

    # 3) HSV: saturation/value + slight hue shift
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_mul = random.uniform(*profile.sat_mul_range)
    val_mul = random.uniform(*profile.val_mul_range)
    hsv[..., 1] *= sat_mul
    hsv[..., 2] *= val_mul
    hue_shift = random.uniform(*profile.hue_shift_range)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4) small BGR cast
    cast = np.array([
        random.uniform(*profile.color_cast_range),
        random.uniform(*profile.color_cast_range),
        random.uniform(*profile.color_cast_range),
    ], dtype=np.float32)
    img = np.clip(img.astype(np.float32) + cast, 0, 255).astype(np.uint8)

    # 5) blur
    sigma = random.uniform(*profile.gaussian_sigma_range)
    k = int(2 * round(3 * sigma) + 1)
    img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

    if random.random() < profile.motion_blur_prob:
        k2 = random.choice(profile.motion_blur_kernels)
        angle = random.uniform(0, np.pi)
        img = _motion_blur(img, k2, angle)

    # 6) chroma smear (video-ish)
    if random.random() < profile.chroma_smear_prob:
        cs = random.uniform(*profile.chroma_smear_sigma_range)
        img = _chroma_smear(img, cs)

    # 7) sensor noise
    noise_sigma = random.uniform(*profile.noise_sigma_range)
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 8) vignette (optional)
    if random.random() < profile.vignette_prob:
        strength = random.uniform(*profile.vignette_strength_range)
        img = _vignette(img, strength)

    # 9) compression artifacts
    q = random.randint(*profile.jpeg_quality_range)
    img = _jpeg_recompress(img, q)

    return img


# ----------------------------
# Batch folder processing
# ----------------------------
def process_folder(
    src_dir: str,
    dst_dir: str,
    variants_per_image: int = 3,
    profile_name: str = "mixed",
    seed: int = 123,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
):
    os.makedirs(dst_dir, exist_ok=True)

    profiles = {
        "daylight": DAYLIGHT,
        "night": NIGHT,
        "ir": IR,
    }

    rng = random.Random(seed)

    def pick_profile():
        if profile_name == "mixed":
            # weighted mix; adjust weights to match your deployment
            r = rng.random()
            return DAYLIGHT if r < 0.55 else NIGHT if r < 0.90 else IR
        return profiles[profile_name]

    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        out_root = os.path.join(dst_dir, rel)
        os.makedirs(out_root, exist_ok=True)

        for fn in files:
            if not fn.lower().endswith(exts):
                continue

            in_path = os.path.join(root, fn)
            img = cv2.imread(in_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            base, _ = os.path.splitext(fn)
            for i in range(variants_per_image):
                prof = pick_profile()
                # make deterministic per image + variant
                local_seed = hash((seed, rel, fn, i, prof.name)) & 0xFFFFFFFF
                out = cctv_augment(img, prof, seed=local_seed)
                out_name = f"{base}__cctv_{prof.name}_{i:02d}.jpg"
                out_path = os.path.join(out_root, out_name)
                cv2.imwrite(out_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])  # save "clean"; artifacts are inside already

    print(f"Done. Output saved to: {dst_dir}")