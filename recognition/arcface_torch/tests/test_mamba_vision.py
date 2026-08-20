"""MambaVision backbone (port of NVlabs/MambaVision adapted to 112x112 faces)."""
import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import pytest


def _tiny(**kw):
    from backbones.mamba_vision import MambaVision
    args = dict(dim=16, in_dim=8, depths=[1, 1, 2, 2], num_heads=[1, 2, 4, 8],
                window_size=[7, 7, 7, 4], mlp_ratio=2, num_features=32)
    args.update(kw)
    return MambaVision(**args)


# ---- architecture: this is what separates the real MambaVision from a flat Mamba stack ----

def test_hierarchy_token_grids_at_112():
    """Stage 3 must see a 7x7 grid and stage 4 a 4x4 grid (stride 16 / 32)."""
    m = _tiny().eval()
    seen = {}
    for i, lvl in enumerate(m.levels):
        lvl.register_forward_hook(lambda mod, inp, out, i=i: seen.__setitem__(i, tuple(inp[0].shape[-2:])))
    with torch.no_grad():
        m(torch.randn(1, 3, 112, 112))
    assert seen == {0: (28, 28), 1: (14, 14), 2: (7, 7), 3: (4, 4)}, seen


def test_stage_composition_conv_then_mixer_then_attention():
    from backbones.mamba_vision import ConvBlock, MambaVisionMixer, Attention
    m = _tiny(depths=[1, 1, 4, 3])
    assert all(isinstance(b, ConvBlock) for b in m.levels[0].blocks)
    assert all(isinstance(b, ConvBlock) for b in m.levels[1].blocks)
    kinds = [type(b.mixer) for b in m.levels[2].blocks]
    assert kinds == [MambaVisionMixer, MambaVisionMixer, Attention, Attention], kinds
    kinds = [type(b.mixer) for b in m.levels[3].blocks]   # odd depth: depth//2+1 mixers
    assert kinds == [MambaVisionMixer, MambaVisionMixer, Attention], kinds


def test_mixer_is_half_channel_ssm_with_expand_1_dstate_8():
    from backbones.mamba_vision import MambaVisionMixer
    mx = MambaVisionMixer(d_model=64)
    assert mx.in_proj.out_features == 64                 # expand = 1
    assert tuple(mx.A_log.shape) == (32, 8)              # SSM on d_inner//2 channels, d_state 8
    y = mx(torch.randn(2, 49, 64))
    assert y.shape == (2, 49, 64)


def test_no_window_padding_at_112():
    """window sizes must divide the 7x7 / 4x4 grids so no zero-padding is computed."""
    m = _tiny()
    assert m.levels[2].window_size == 7 and m.levels[3].window_size == 4


# ---- selective scan fallback matches the textbook recurrence ----

def test_selective_scan_reference_matches_naive_loop():
    from backbones.mamba_vision import selective_scan_ref
    torch.manual_seed(0)
    Bsz, D, N, L = 2, 6, 4, 9
    u, delta = torch.randn(Bsz, D, L), torch.randn(Bsz, D, L)
    A = -torch.rand(D, N) - 0.1
    Bm, Cm = torch.randn(Bsz, N, L), torch.randn(Bsz, N, L)
    Dp, bias = torch.randn(D), torch.randn(D)
    y = selective_scan_ref(u, delta, A, Bm, Cm, Dp, delta_bias=bias, delta_softplus=True)
    dlt = F.softplus(delta + bias[None, :, None])
    h = torch.zeros(Bsz, D, N); ys = []
    for t in range(L):
        h = torch.exp(dlt[:, :, t, None] * A) * h + dlt[:, :, t, None] * Bm[:, None, :, t] * u[:, :, t, None]
        ys.append((h * Cm[:, None, :, t]).sum(-1) + Dp * u[:, :, t])
    assert torch.allclose(y, torch.stack(ys, -1), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_selective_scan_kernel_matches_reference_if_installed():
    from backbones.mamba_vision import selective_scan_ref, selective_scan_fn
    if selective_scan_fn is selective_scan_ref:
        pytest.skip("mamba_ssm not installed")
    torch.manual_seed(0)
    u, delta = torch.randn(2, 8, 49, device="cuda"), torch.randn(2, 8, 49, device="cuda")
    A = -torch.rand(8, 8, device="cuda") - 0.1
    Bm, Cm = torch.randn(2, 8, 49, device="cuda"), torch.randn(2, 8, 49, device="cuda")
    Dp, bias = torch.randn(8, device="cuda"), torch.randn(8, device="cuda")
    y_k = selective_scan_fn(u, delta, A, Bm, Cm, Dp, z=None, delta_bias=bias, delta_softplus=True)
    y_r = selective_scan_ref(u, delta, A, Bm, Cm, Dp, delta_bias=bias, delta_softplus=True)
    assert torch.allclose(y_k, y_r, atol=1e-4)


# ---- shared backbone contract used by train_v2 / train_adaface ----

def test_get_model_sizes_and_embedding_shape():
    from backbones import get_model
    for name, dim in [("mambavision_t", 80), ("mambavision_s", 96), ("mambavision_b", 128), ("mambavision_l", 196)]:
        m = get_model(name, num_features=512, fp16=False).eval()
        assert m.levels[0].blocks[0].conv1.in_channels == dim, name
        with torch.no_grad():
            out = m(torch.randn(2, 3, 112, 112))
        assert out.shape == (2, 512), name


def test_norm_output_returns_unit_embedding_and_norm():
    from backbones import get_model
    m = get_model("mambavision_t", num_features=512, fp16=False, norm_output=True).train()
    emb, norm = m(torch.randn(2, 3, 112, 112))
    assert emb.shape == (2, 512) and norm.shape == (2, 1)
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-4)


def test_backward_reaches_ssm_attention_and_conv():
    m = _tiny().train()
    out = m(torch.randn(2, 3, 112, 112))
    # NB: the head ends in BatchNorm1d, so out.sum() is constant (N*beta) in train mode;
    # use a loss that actually depends on the embedding direction.
    weight = F.normalize(torch.randn(10, out.shape[1]), dim=1)
    F.cross_entropy(F.normalize(out, dim=1) @ weight.T * 16, torch.arange(2)).backward()
    lvl2 = m.levels[2].blocks
    assert lvl2[0].mixer.A_log.grad is not None and lvl2[0].mixer.A_log.grad.abs().sum() > 0
    assert lvl2[-1].mixer.qkv.weight.grad is not None
    assert m.levels[0].blocks[0].conv1.weight.grad is not None


def test_eval_deterministic():
    m = _tiny().eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        assert torch.equal(m(x), m(x))
