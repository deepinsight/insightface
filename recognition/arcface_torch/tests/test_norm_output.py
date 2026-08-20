"""norm_output=True must make every backbone family return (unit-norm embedding, norm)
so that train_adaface.py / train_transface.py --loss adaface work with all of them."""
import torch

from backbones import get_model
from backbones.mamba_vision import MambaVision
from backbones.mobilefacenet import get_mbf
from backbones.transface_vit import TransFaceViT
from backbones.vit import VisionTransformer


def _check(out, batch):
    emb, norm = out[0], out[1]
    assert emb.shape == (batch, 512)
    assert norm.shape == (batch, 1)
    assert torch.allclose(emb.norm(dim=1), torch.ones(batch), atol=1e-4)
    assert (norm > 0).all()


def _small_vit(cls, **kw):
    return cls(img_size=112, patch_size=9, num_classes=512, embed_dim=64, depth=2,
               num_heads=4, drop_path_rate=0.0, norm_layer="ln", mask_ratio=0.1, **kw)


def test_mbf_norm_output():
    m = get_mbf(fp16=False, num_features=512, norm_output=True).train()
    _check(m(torch.randn(2, 3, 112, 112)), 2)


def test_vit_norm_output():
    m = _small_vit(VisionTransformer, norm_output=True).train()
    _check(m(torch.randn(2, 3, 112, 112)), 2)


def test_mamba_norm_output():
    m = MambaVision(dim=16, in_dim=8, depths=[1, 1, 2, 2], num_heads=[1, 2, 4, 8],
                    window_size=[7, 7, 7, 4], mlp_ratio=2, num_features=512,
                    norm_output=True).train()
    _check(m(torch.randn(2, 3, 112, 112)), 2)


def test_transface_norm_output_train_and_eval():
    m = _small_vit(TransFaceViT, norm_output=True).train()
    out = m(torch.randn(2, 3, 112, 112))
    assert len(out) == 4, "train: (emb, norm, patch_weight, patch_entropy)"
    _check(out, 2)
    assert out[2].shape == (2, 144) and out[3].shape == (2, 144)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 112, 112))
    assert len(out) == 2, "eval: (emb, norm)"
    _check(out, 2)


def test_default_is_plain_embedding():
    for name in ("mbf", "vit_t", "mambavision_t"):
        m = get_model(name, fp16=False, num_features=512).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 112, 112))
        assert isinstance(out, torch.Tensor) and out.shape == (2, 512), name
