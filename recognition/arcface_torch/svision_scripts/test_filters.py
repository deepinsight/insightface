import cv2
import numpy as np
import random

def cctv_degrade(bgr):
    h, w = bgr.shape[:2]

    # 1) Downscale -> upscale (resolution loss)
    scale = random.uniform(0.25, 0.45)
    small = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2) Gamma + exposure (darker)
    gamma = random.uniform(1.2, 1.8)  # >1 darkens midtones
    lut = np.array([((i/255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, lut)

    # 3) Reduce saturation + add slight color cast
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= random.uniform(0.6, 0.85)  # desaturate
    hsv[...,2] *= random.uniform(0.85, 1.05) # small brightness tweak
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cast = np.array([random.uniform(-8, 8), random.uniform(-6, 10), random.uniform(-8, 8)], dtype=np.float32)
    img = np.clip(img.astype(np.float32) + cast, 0, 255).astype(np.uint8)

    # 4) Blur (Gaussian)
    if random.random() < 0.9:
        sigma = random.uniform(0.6, 1.4)
        k = int(2 * round(3*sigma) + 1)
        img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

    # 5) Optional motion blur
    if random.random() < 0.35:
        k = random.choice([3,5,7,9])
        angle = random.uniform(0, np.pi)
        kernel = np.zeros((k, k), dtype=np.float32)
        cx = cy = k // 2
        for i in range(k):
            x = int(cx + (i - cx) * np.cos(angle))
            y = int(cy + (i - cy) * np.sin(angle))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        kernel /= kernel.sum() if kernel.sum() > 0 else 1.0
        img = cv2.filter2D(img, -1, kernel)

    # 6) Add sensor noise
    if random.random() < 0.9:
        noise_sigma = random.uniform(3, 12)
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 7) JPEG compression artifacts
    q = random.randint(15, 45)
    enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img