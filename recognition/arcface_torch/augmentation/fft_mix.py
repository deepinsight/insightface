"""DPAP - Dominant Patch Amplitude Perturbation (TransFace, ICCV 2023).

For a fraction `prob` of the images in a batch, the `top_k` patches with the
largest SE weights (the "dominant" patches the ViT over-relies on) get their
Fourier amplitude spectrum mixed with that of a random patch from a random
image of the batch, lam ~ U(0, alpha), while their phase is kept. This is the
official per-patch numpy procedure (TransFace/train.py + FFT.py) rewritten as
batched tensor ops so it runs on the GPU.
"""
import torch


def amplitude_mix(src: torch.Tensor, ref: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Mix the full amplitude spectrum of `ref` into `src`, keeping src's phase.

    src, ref: (N, C, h, w) float;  lam: (N,) mixing coefficients in [0, 1].
    out = Re(IFFT( ((1 - lam) |S| + lam |R|) * exp(i * angle(S)) ))
    """
    S = torch.fft.fft2(src.float())
    R = torch.fft.fft2(ref.float())
    lam = lam.float().view(-1, 1, 1, 1)
    amp = (1 - lam) * S.abs() + lam * R.abs()
    return torch.fft.ifft2(torch.polar(amp, S.angle())).real


def dpap_perturb(img: torch.Tensor, patch_weight: torch.Tensor, top_k: int = 7, prob: float = 0.2,
                 alpha: float = 1.0, patch_size: int = 9) -> torch.Tensor:
    """Apply DPAP to a normalised image batch.

    img:          (B, C, H, W), values as produced by the dataloader (x/255 - 0.5) / 0.5
    patch_weight: (B, P) SE patch weights from TransFaceViT (P = (H/patch_size) * (W/patch_size))
    Returns a new tensor; un-perturbed pixels are bit-identical to the input. Perturbed patches
    are clipped to [0, 255] and truncated to the uint8 grid like the original implementation.
    Only the region covered by the patch grid (H//ps*ps x W//ps*ps, i.e. 108x108 for 112/9) is
    touched - the ViT patch embedding never sees the remaining border either.
    """
    B, C, H, W = img.shape
    ps = patch_size
    gh, gw = H // ps, W // ps
    P = gh * gw
    if prob <= 0 or top_k <= 0:
        return img
    sel = torch.nonzero(torch.rand(B, device=img.device) < prob).flatten()
    if sel.numel() == 0:
        return img
    k = min(top_k, P)
    Hc, Wc = gh * ps, gw * ps
    # (B, P, C, ps, ps) patch view of the 0-255 pixel values inside the patch grid
    pix = (img[:, :, :Hc, :Wc] * 0.5 + 0.5) * 255
    patches = pix.reshape(B, C, gh, ps, gw, ps).permute(0, 2, 4, 1, 3, 5).reshape(B, P, C, ps, ps)
    top = torch.topk(patch_weight[sel], k=k, dim=1).indices                       # (n, k)
    n = sel.numel()
    src = patches[sel[:, None], top]                                                # (n, k, C, ps, ps)
    ref_img = torch.randint(0, B, (n, k), device=img.device)
    ref_patch = torch.randint(0, P, (n, k), device=img.device)
    ref = patches[ref_img, ref_patch]
    lam = torch.rand(n * k, device=img.device) * alpha
    mixed = amplitude_mix(src.reshape(n * k, C, ps, ps), ref.reshape(n * k, C, ps, ps), lam)
    mixed = mixed.clamp(0, 255).floor().reshape(n, k, C, ps, ps)
    mixed = ((mixed / 255) - 0.5) / 0.5                                             # back to normalised space
    grid = img[:, :, :Hc, :Wc].reshape(B, C, gh, ps, gw, ps).permute(0, 2, 4, 1, 3, 5).reshape(B, P, C, ps, ps).clone()
    grid[sel[:, None], top] = mixed.to(grid.dtype)
    out = img.clone()
    out[:, :, :Hc, :Wc] = grid.reshape(B, gh, gw, C, ps, ps).permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hc, Wc)
    return out
