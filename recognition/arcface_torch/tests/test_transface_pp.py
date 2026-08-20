"""TransFace++ (TPAMI 2025): byte-level ViT with topology features, TIBC and EHSM."""
import io
import os
import sys
import tempfile
sys.path.insert(0, ".")
import numpy as np
import pytest
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# image -> bytes
# --------------------------------------------------------------------------- #
def _norm_img(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    pix = torch.randint(0, 256, (b, 3, 112, 112), generator=g).float()
    return (pix / 255 - 0.5) / 0.5, pix.to(torch.uint8)          # normalised (as the dataloader emits), raw uint8


def test_image_to_bytes_fhwc_is_hwc_order():
    from backbones.transface_pp import image_to_bytes
    img, pix = _norm_img()
    b = image_to_bytes(img, "fhwc")
    assert b.dtype == torch.long and b.shape == (2, 112 * 112 * 3)
    assert torch.equal(b.view(2, 112, 112, 3), pix.permute(0, 2, 3, 1).long())


def test_image_to_bytes_fchw_is_chw_order():
    from backbones.transface_pp import image_to_bytes
    img, pix = _norm_img()
    b = image_to_bytes(img, "fchw")
    assert torch.equal(b.view(2, 3, 112, 112), pix.long())


@pytest.mark.parametrize("fmt,expected_len", [("png", 37817), ("tiff", 37772)])
def test_image_to_bytes_encoded_formats_roundtrip(fmt, expected_len):
    """PNG (stored, level 0) / TIFF (uncompressed) bytes have a constant length and decode losslessly."""
    from PIL import Image
    from backbones.transface_pp import image_to_bytes, num_bytes_for
    img, pix = _norm_img(b=3, seed=1)
    b = image_to_bytes(img, fmt)
    assert b.shape == (3, expected_len) and num_bytes_for(fmt) == expected_len
    for i in range(3):
        decoded = np.array(Image.open(io.BytesIO(bytes(b[i].tolist()))))
        assert np.array_equal(decoded, pix[i].permute(1, 2, 0).numpy())


def test_num_bytes_for_raw_formats():
    from backbones.transface_pp import num_bytes_for
    assert num_bytes_for("fhwc") == num_bytes_for("fchw") == 37632


# --------------------------------------------------------------------------- #
# bytes projector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_bytes,k2", [(37632, 63), (37817, 74), (37772, 71)])
def test_byte_tokenizer_yields_144_tokens(num_bytes, k2):
    from backbones.transface_pp import ByteTokenizer
    tok = ByteTokenizer(num_bytes=num_bytes, embed_dim=64)
    assert tok.conv2.kernel_size[0] == k2
    x = torch.randint(0, 256, (2, num_bytes))
    out = tok(x)
    assert out.shape == (2, 144, 64)


def test_byte_tokenizer_has_256_entry_vocab():
    from backbones.transface_pp import ByteTokenizer
    tok = ByteTokenizer(num_bytes=37632, embed_dim=64)
    assert tok.embeddings.num_embeddings == 256 and tok.embeddings.embedding_dim == 128


# --------------------------------------------------------------------------- #
# persistent homology
# --------------------------------------------------------------------------- #
def _kruskal_mst_lengths(pts):
    n = len(pts)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    edges = sorted((d[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]; u = parent[u]
        return u
    out = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj; out.append(w)
    return np.array(out)


def test_mst_persistence_matches_kruskal_and_is_differentiable():
    from backbones.transface_pp import mst_persistence
    torch.manual_seed(0)
    pts = torch.randn(3, 20, 3, requires_grad=True)
    pers = mst_persistence(pts)
    assert pers.shape == (3, 19)
    for b in range(3):
        ref = _kruskal_mst_lengths(pts[b].detach().numpy())
        assert np.allclose(np.sort(pers[b].detach().numpy()), ref, atol=1e-5)
        assert torch.all(pers[b][1:] >= pers[b][:-1]), "ascending like the union-find scan"
    pers.sum().backward()
    assert pts.grad is not None and pts.grad.abs().sum() > 0


def _ref_sublevel_persistence(f):
    """Union-find 0-dim sublevel persistence with gda-public conventions:
    elder rule on (value, index); the essential class dies at the last vertex processed."""
    n = len(f)
    order = sorted(range(n), key=lambda i: (f[i], i))
    parent, cmin, pairs = {}, {}, []
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]; u = parent[u]
        return u
    def key(i): return (f[i], i)
    for v in order:
        parent[v] = v; cmin[v] = v
        roots = {find(u) for u in (v - 1, v + 1) if 0 <= u < n and u in parent}
        roots = sorted(roots, key=lambda r: key(cmin[r]))          # elder first
        if len(roots) == 2:
            younger = cmin[roots[1]]
            pairs.append((younger, v, f[v] - f[younger]))
        for r in roots:
            parent[r] = parent[v] if r != roots[0] else parent[r]
        if roots:
            elder = roots[0]
            for r in roots + [v]:
                parent[find(r)] = elder
            cmin[elder] = min((cmin[r] for r in roots + [v]), key=key)
    gmin, last = order[0], order[-1]
    pairs.append((gmin, last, f[last] - f[gmin]))
    return sorted(pairs)


def test_sublevel_persistence_matches_reference_with_ties():
    from backbones.transface_pp import sublevel_persistence
    assert _ref_sublevel_persistence([2.0, 3.0, 0.0, 5.0, 2.5, 2.9]) == [
        (0, 1, 1.0), (2, 3, 5.0), (4, 3, 2.5)]                      # the gda-public docstring example
    g = torch.Generator().manual_seed(0)
    sig = torch.randint(0, 6, (4, 60), generator=g).float()         # byte-like, many plateaus
    birth, death, pers = sublevel_persistence(sig)
    assert birth.shape == death.shape == pers.shape and birth.shape[0] == 4
    for b in range(4):
        got = sorted((int(i), int(j), round(float(p), 5)) for i, j, p in zip(birth[b], death[b], pers[b]) if i >= 0)
        ref = [(i, j, round(p, 5)) for i, j, p in _ref_sublevel_persistence(sig[b].tolist())]
        assert got == ref, (b, got[:5], ref[:5])


def _ref_tibc_indices(f, num_keep):
    """compress_tsc semantics: endpoints + (birth, death) indices of the most persistent pairs,
    extended until `num_keep` unique indices are collected."""
    pairs = sorted(_ref_sublevel_persistence(f), key=lambda t: (-t[2], t[0]))
    ordered = [0, len(f) - 1] + [i for p in pairs for i in (p[0], p[1])]
    if len(set(ordered)) <= num_keep:
        return sorted(set(ordered))
    k = num_keep
    while len(set(ordered[:k])) < num_keep:
        k += 1
    return sorted(set(ordered[:k]))


def test_tibc_compress_keeps_endpoints_and_most_persistent_points():
    from backbones.transface_pp import tibc_compress
    g = torch.Generator().manual_seed(1)
    sig = torch.rand(3, 300, generator=g) * 255                        # continuous -> no persistence ties
    vals, idx = tibc_compress(sig, num_keep=16, return_indices=True)
    assert vals.shape == (3, 16) and idx.shape == (3, 16)
    for b in range(3):
        ref = _ref_tibc_indices(sig[b].tolist(), 16)
        assert idx[b].tolist() == ref
        assert torch.allclose(vals[b], F.normalize(sig[b][idx[b]], dim=0), atol=1e-5), "values are L2-normalised"


def test_tibc_compress_pads_when_signal_has_few_critical_points():
    from backbones.transface_pp import tibc_compress
    sig = torch.linspace(0, 255, 50).unsqueeze(0)                      # monotone: only the endpoints survive
    vals, idx = tibc_compress(sig, num_keep=8, return_indices=True)
    assert idx[0].tolist()[:2] == [0, 49] and (idx[0][2:] == -1).all()
    assert (vals[0][2:] == 0).all()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _tiny(**kw):
    from backbones.transface_pp import TransFacePPViT
    args = dict(num_bytes=37632, num_classes=32, embed_dim=64, depth=2, num_heads=4, mask_ratio=0.1)
    args.update(kw)
    return TransFacePPViT(**args)


def test_model_train_returns_embedding_and_patch_entropy():
    m = _tiny().train()
    b = torch.randint(0, 256, (2, 37632))
    emb, ent = m(b)
    assert emb.shape == (2, 32) and ent.shape == (2, 144)
    assert (ent >= 0).all()


def test_model_eval_returns_embedding_only_and_is_deterministic():
    m = _tiny().eval()
    b = torch.randint(0, 256, (2, 37632))
    with torch.no_grad():
        out = m(b)
        assert isinstance(out, torch.Tensor) and out.shape == (2, 32)
        assert torch.equal(out, m(b))


def test_model_norm_output_contract():
    m = _tiny(norm_output=True).train()
    emb, norm, ent = m(torch.randint(0, 256, (2, 37632)))
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-4) and norm.shape == (2, 1)
    m.eval()
    with torch.no_grad():
        emb, norm = m(torch.randint(0, 256, (2, 37632)))
    assert emb.shape == (2, 32) and norm.shape == (2, 1)


def test_topology_features_enter_last_block_only():
    from backbones.transface_pp import TopologyFeature
    m = _tiny(depth=3)
    assert isinstance(m.topology, TopologyFeature)
    assert [blk.use_topology for blk in m.blocks] == [False, False, True]
    m2 = _tiny(use_topology=False)
    assert m2.topology is None


def test_backward_reaches_byte_embedding_topology_and_senet():
    m = _tiny(tibc_prob=1.0).train()
    emb, _ = m(torch.randint(0, 256, (2, 37632)))
    weight = F.normalize(torch.randn(10, emb.shape[1]), dim=1)
    F.cross_entropy(F.normalize(emb, dim=1) @ weight.T * 16, torch.arange(2)).backward()
    assert m.tokenizer.embeddings.weight.grad.abs().sum() > 0
    assert m.topology.increase[0].weight.grad.abs().sum() > 0
    assert m.senet[0].weight.grad.abs().sum() > 0


def test_get_model_names_and_byte_format():
    from backbones import get_model
    for name in ("transface_pp_vit_s", "transface_pp_vit_b"):
        m = get_model(name, num_features=512, fp16=False, byte_format="fhwc").eval()
        assert m.num_bytes == 37632
    m = get_model("transface_pp_vit_s", num_features=512, fp16=False, byte_format="png")
    assert m.num_bytes == 37817 and m.tokenizer.conv2.kernel_size[0] == 74


# --------------------------------------------------------------------------- #
# EHSM: entropy-guided hard sample mining
# --------------------------------------------------------------------------- #
def test_ehsm_sample_weight_range_and_monotonicity():
    from losses import ehsm_sample_weight
    ent = torch.tensor([[0.0] * 144, [1.0] * 144, [10.0] * 144])
    w = ehsm_sample_weight(ent, gamma=1.0)
    assert w.shape == (3, 1)
    assert torch.allclose(w.view(-1), torch.tensor([2.0, 1 + np.exp(-1.0), 1 + np.exp(-10.0)], dtype=torch.float32), atol=1e-5)
    assert (w[0] > w[1] > w[2]).all(), "low-entropy (hard) samples get larger weight"


def _init_single_process_group():
    from torch import distributed
    if not distributed.is_initialized():
        f = os.path.join(tempfile.gettempdir(), f"pfc_test_{os.getpid()}")
        distributed.init_process_group("gloo", init_method=f"file://{f}", rank=0, world_size=1)


def test_dist_cross_entropy_accepts_sample_weight():
    from partial_fc_v2 import DistCrossEntropy
    _init_single_process_group()
    torch.manual_seed(0)
    logits = torch.randn(4, 7)
    labels = torch.tensor([[1], [3], [5], [6]])                         # single rank: every label is local
    w = torch.tensor([[1.5], [1.0], [2.0], [0.5]])
    l1 = logits.clone().requires_grad_(); l2 = logits.clone().requires_grad_()
    loss = DistCrossEntropy()(l1, labels, w)
    ce = F.cross_entropy(l2, labels.view(-1), reduction="none")
    ref = (ce * w.view(-1)).mean()
    assert torch.allclose(loss, ref, atol=1e-6)
    loss.backward(); ref.backward()
    assert torch.allclose(l1.grad, l2.grad, atol=1e-6)
    assert torch.allclose(DistCrossEntropy()(logits.clone(), labels), DistCrossEntropy()(logits.clone(), labels, torch.ones(4, 1)))
