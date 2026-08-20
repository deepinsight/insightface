"""TransFace (ICCV 2023) as in the paper / official code: SE patch weights, DPAP (dominant patch
amplitude perturbation, patch-level FFT mixing on the GPU) and patch entropy for EHSM."""
import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import pytest


def _small_vit(**kw):
    from backbones.transface_vit import TransFaceViT
    args = dict(img_size=112, patch_size=9, num_classes=512, embed_dim=64, depth=2,
                num_heads=4, drop_path_rate=0.0, norm_layer="ln", mask_ratio=0.0)
    args.update(kw)
    return TransFaceViT(**args)


# ---- DPAP ------------------------------------------------------------------
def test_amplitude_mix_lam_zero_is_identity_and_lam_one_takes_ref_amplitude():
    from augmentation.fft_mix import amplitude_mix
    torch.manual_seed(0)
    src = torch.rand(4, 3, 9, 9)
    assert torch.allclose(amplitude_mix(src, torch.rand(4, 3, 9, 9), torch.zeros(4)), src, atol=1e-5)
    ref = 2 * src                                    # same phase, double amplitude
    assert torch.allclose(amplitude_mix(src, ref, torch.ones(4)), ref, atol=1e-4)
    half = amplitude_mix(src, ref, torch.full((4,), 0.5))
    assert torch.allclose(half, 1.5 * src, atol=1e-4), "amplitude is (1-lam)|S| + lam|R|, phase from src"


def test_dpap_prob_zero_is_identity():
    from augmentation.fft_mix import dpap_perturb
    torch.manual_seed(0)
    img = torch.rand(3, 3, 112, 112) * 2 - 1
    w = torch.rand(3, 144).softmax(1)
    assert torch.equal(dpap_perturb(img, w, top_k=7, prob=0.0, alpha=1.0), img)


def test_dpap_changes_only_the_top_k_patches_of_selected_images():
    from augmentation.fft_mix import dpap_perturb
    torch.manual_seed(0)
    pix = torch.randint(0, 256, (4, 3, 112, 112)).float()
    img = (pix / 255 - 0.5) / 0.5                    # dataloader normalisation
    w = torch.rand(4, 144).softmax(1)
    out = dpap_perturb(img, w, top_k=7, prob=1.0, alpha=1.0)
    diff = (out != img)
    assert not diff[:, :, 108:, :].any() and not diff[:, :, :, 108:].any(), "border outside the patch grid untouched"
    changed = diff[:, :, :108, :108].reshape(4, 3, 12, 9, 12, 9).any(dim=(1, 3, 5)).reshape(4, 144)   # per 9x9 patch
    top = torch.topk(w, 7, dim=1).indices
    for b in range(4):
        assert set(changed[b].nonzero().flatten().tolist()) <= set(top[b].tolist())
        assert changed[b].sum() >= 5, "with lam ~ U(0, 1) almost every dominant patch is perturbed"
    # perturbed pixels are quantised to the uint8 grid and stay in range, like the original numpy code
    back = (out * 0.5 + 0.5) * 255
    assert torch.allclose(back, back.round(), atol=1e-4) and back.min() >= 0 and back.max() <= 255


def test_dpap_prob_selects_a_subset_of_images():
    from augmentation.fft_mix import dpap_perturb
    torch.manual_seed(0)
    img = torch.rand(64, 3, 112, 112) * 2 - 1
    w = torch.rand(64, 144).softmax(1)
    out = dpap_perturb(img, w, top_k=7, prob=0.2, alpha=1.0)
    n_changed = (out != img).flatten(1).any(1).sum().item()
    assert 3 <= n_changed <= 30, n_changed


# ---- backbone ---------------------------------------------------------------
def test_transface_vit_train_returns_embedding_patch_weight_and_entropy():
    m = _small_vit().train()
    emb, weight, entropy = m(torch.randn(2, 3, 112, 112))
    assert emb.shape == (2, 512)
    assert weight.shape == (2, 144) and torch.allclose(weight.sum(1), torch.ones(2), atol=1e-5)
    assert entropy.shape == (2, 144) and (entropy >= 0).all()


def test_transface_vit_has_se_module_and_uses_it():
    m = _small_vit()
    assert hasattr(m, "senet") and isinstance(m.senet[-1], torch.nn.Sigmoid)
    m.train()
    x = torch.randn(2, 3, 112, 112)
    emb1, _, _ = m(x)
    for p in m.senet.parameters():
        p.data.zero_()                                # all patch gates become sigmoid(0)=0.5
    emb2, _, _ = m(x)
    assert not torch.allclose(emb1, emb2), "SE gates must modulate the patch features"


def test_transface_vit_eval_returns_tensor():
    m = _small_vit().eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 112, 112))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 512)


def test_transface_vit_norm_output_contract():
    m = _small_vit(norm_output=True).train()
    emb, norm, weight, entropy = m(torch.randn(2, 3, 112, 112))
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-4) and norm.shape == (2, 1)
    m.eval()
    with torch.no_grad():
        emb, norm = m(torch.randn(2, 3, 112, 112))
    assert emb.shape == (2, 512) and norm.shape == (2, 1)


def test_transface_vit_l_shape():
    from backbones import get_model
    m = get_model("transface_vit_l", num_features=512).train()
    emb, weight, entropy = m(torch.randn(2, 3, 112, 112))
    assert emb.shape == (2, 512) and weight.shape == (2, 144) and entropy.shape == (2, 144)
