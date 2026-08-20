"""TransFace++ backbone: a ViT that recognises faces directly from image bytes.

Port of the TransFace++ model (Dan et al., "TransFace++: Rethinking the Face
Recognition Paradigm with a Focus on Accuracy, Efficiency, and Security",
IEEE TPAMI 2025; official code https://github.com/DanJun6737/TransFace_pp).

Pipeline (train):
    image bytes  B = (b_1..b_n)            fHWC / fCHW raw pixels, PNG or TIFF file bytes
      -> bytes projector  H                256-entry byte embedding + 2 strided Conv1d -> 144 tokens
      -> topology feature                  Conv1d -> 20 points -> 0-dim persistence (MST) -> MLP -> (144, D)
      -> TIBC                              topological byte compression of B added to the tokens (p = 0.3)
      -> ViT blocks                        topology feature cross-attended (as K/V) in the last block only
      -> SE patch re-weighting -> embedding head
    returns (embedding, patch_entropy) where patch_entropy (B, 144) is the per-token feature std
    used by EHSM (entropy-guided hard sample mining) in the loss.

Everything (including both persistence computations) runs batched on the GPU
with no external topology dependencies.
"""
import io
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

BYTE_FORMATS = ("fhwc", "fchw", "png", "tiff")
_PIL_KWARGS = {"png": dict(format="PNG", compress_level=0), "tiff": dict(format="TIFF")}


# --------------------------------------------------------------------------- #
# image -> bytes
# --------------------------------------------------------------------------- #
def _denormalise_uint8(img: torch.Tensor) -> torch.Tensor:
    """Inverse of the dataloader's Normalize(0.5, 0.5): (B, 3, H, W) in [-1, 1] -> uint8."""
    return ((img.float() * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8)


def _encode_with_pil(hwc: np.ndarray, fmt: str) -> np.ndarray:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(hwc).save(buf, **_PIL_KWARGS[fmt])
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


_NUM_BYTES_CACHE = {}


def num_bytes_for(byte_format: str, image_size: int = 112, channels: int = 3) -> int:
    """Length of the byte sequence a (image_size x image_size x channels) image produces."""
    assert byte_format in BYTE_FORMATS, byte_format
    if byte_format in ("fhwc", "fchw"):
        return image_size * image_size * channels
    key = (byte_format, image_size, channels)
    if key not in _NUM_BYTES_CACHE:
        dummy = np.zeros((image_size, image_size, channels), dtype=np.uint8)
        _NUM_BYTES_CACHE[key] = int(_encode_with_pil(dummy, byte_format).shape[0])
    return _NUM_BYTES_CACHE[key]


def image_to_bytes(img: torch.Tensor, byte_format: str = "tiff", num_bytes: Optional[int] = None) -> torch.Tensor:
    """Convert a normalised image batch (B, 3, H, W) to its byte sequence (B, L) as int64.

    fhwc / fchw: raw uint8 pixels in H-W-C / C-H-W order (no decoding cost, pure GPU).
    png / tiff:  file bytes identical to PIL's (PNG stored/level 0, TIFF uncompressed), built on the GPU.
    """
    assert byte_format in BYTE_FORMATS, byte_format
    u8 = _denormalise_uint8(img)
    if byte_format == "fhwc":
        out = u8.permute(0, 2, 3, 1).reshape(u8.shape[0], -1).long()
    elif byte_format == "fchw":
        out = u8.reshape(u8.shape[0], -1).long()
    else:
        # byte-exact PIL-compatible TIFF / PNG writers, batched on the images' device
        from .byte_codecs import encode_image_bytes
        out = encode_image_bytes(u8.permute(0, 2, 3, 1).contiguous(), byte_format)
    if num_bytes is not None and out.shape[1] != num_bytes:
        out = F.pad(out, (0, max(0, num_bytes - out.shape[1])))[:, :num_bytes]
    return out


# --------------------------------------------------------------------------- #
# bytes projector
# --------------------------------------------------------------------------- #
class ByteTokenizer(nn.Module):
    """Byte embedding (vocab 256) + two strided Conv1d layers that reduce any
    byte sequence to `num_tokens` tokens. The second kernel is derived from the
    byte length (63 for 37,632 raw bytes, 74 for PNG, 71 for TIFF at 112x112)."""

    def __init__(self, num_bytes: int, embed_dim: int, num_tokens: int = 144, vocab_size: int = 256,
                 byte_embed_dim: int = 128, mid_channels: int = 256,
                 kernel1: int = 32, stride1: int = 16, stride2: int = 16):
        super().__init__()
        self.num_bytes = num_bytes
        self.num_tokens = num_tokens
        len1 = (num_bytes - kernel1) // stride1 + 1
        kernel2 = len1 - (num_tokens - 1) * stride2
        assert kernel2 >= 1, f"{num_bytes} bytes are too few for {num_tokens} tokens"
        self.embeddings = nn.Embedding(vocab_size, byte_embed_dim)
        nn.init.trunc_normal_(self.embeddings.weight, std=math.sqrt(1.0 / embed_dim))
        self.conv1 = nn.Conv1d(byte_embed_dim, mid_channels, kernel_size=kernel1, stride=stride1, bias=False)
        self.conv2 = nn.Conv1d(mid_channels, embed_dim, kernel_size=kernel2, stride=stride2, bias=False)

    def forward(self, byte_seq: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(byte_seq.long())                     # (B, L, 128)
        x = self.conv2(self.conv1(x.transpose(1, 2)))           # (B, D, num_tokens)
        return x.transpose(1, 2)


# --------------------------------------------------------------------------- #
# persistent homology
# --------------------------------------------------------------------------- #
def mst_persistence(points: torch.Tensor) -> torch.Tensor:
    """0-dimensional Vietoris-Rips persistence of a point cloud (B, n, d).

    The finite death times are exactly the edge lengths of the minimum spanning
    tree; returned ascending, shape (B, n-1). Differentiable w.r.t. the points
    (batched Prim's algorithm, gradient flows through the selected distances).
    """
    B, n, _ = points.shape
    d = torch.cdist(points.float(), points.float(), compute_mode="donot_use_mm_for_euclid_dist")
    ar = torch.arange(B, device=points.device)
    in_tree = F.one_hot(torch.zeros(B, dtype=torch.long, device=points.device), n).bool()
    best = d[:, 0, :]
    lengths = []
    for _ in range(n - 1):
        w, j = best.masked_fill(in_tree, float("inf")).min(dim=1)
        lengths.append(w)
        in_tree = in_tree | F.one_hot(j, n).bool()          # out-of-place: autograd keeps the old mask
        best = torch.minimum(best, d[ar, j, :])
    pers = torch.stack(lengths, dim=1)
    return pers.sort(dim=1).values


def _rank_keys(values: torch.Tensor) -> torch.Tensor:
    """Unique integer keys ordering (value, index) lexicographically, shape (B, n)."""
    B, n = values.shape
    idx = torch.arange(n, device=values.device).expand(B, n)
    order = torch.argsort(values.double(), dim=1, stable=True)   # stable -> ties broken by index
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, idx)
    return ranks


def _sparse_table(g: torch.Tensor, op: str):
    """Sparse table over (B, n) int keys: levels[k][:, i] = op(g[i : i + 2**k]) (invalid -> sentinel).
    Returns (values (B, K+1, n), argpos (B, K+1, n) or None)."""
    B, n = g.shape
    K = max(1, math.ceil(math.log2(n))) if n > 1 else 1
    idx = torch.arange(n, device=g.device).expand(B, n)
    sentinel = -1 if op == "max" else n + 1
    vals, args = [g], [idx]
    for k in range(1, K + 1):
        half = 1 << (k - 1)
        prev_v, prev_a = vals[-1], args[-1]
        shifted_v = torch.full_like(prev_v, sentinel)
        shifted_a = torch.full_like(prev_a, -1)
        if half < n:
            shifted_v[:, :-half] = prev_v[:, half:]
            shifted_a[:, :-half] = prev_a[:, half:]
        if op == "max":
            take_shift = shifted_v > prev_v
        else:
            take_shift = shifted_v < prev_v
        v = torch.where(take_shift, shifted_v, prev_v)
        a = torch.where(take_shift, shifted_a, prev_a)
        valid = idx + (1 << k) <= n
        v = torch.where(valid, v, torch.full_like(v, sentinel))
        vals.append(v); args.append(a)
    return torch.stack(vals, 1), torch.stack(args, 1)


def _range_argmin(min_vals: torch.Tensor, min_args: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """argmin of g over inclusive ranges [a, b] (per element), using the min sparse table."""
    B, n = a.shape
    length = (b - a + 1).clamp(min=1)
    k = torch.floor(torch.log2(length.double())).long()
    pow2 = (1 << k)
    start2 = (b - pow2 + 1).clamp(min=0)
    ar = torch.arange(B, device=a.device)[:, None].expand(B, n)
    v1 = min_vals[ar, k, a.clamp(0, n - 1)]
    v2 = min_vals[ar, k, start2]
    i1 = min_args[ar, k, a.clamp(0, n - 1)]
    i2 = min_args[ar, k, start2]
    return torch.where(v1 <= v2, i1, i2)


def sublevel_persistence(values: torch.Tensor):
    """0-dim sublevel-set persistence of 1-D signals, batched and vectorised on GPU.

    values: (B, n). Ties are broken by index (processing order = (value, index)),
    the elder rule picks the survivor, and - as in gda-public's `Signal.make_pers`
    used by TransFace++ - the essential class is paired with the global maximum.
    Returns (birth_idx, death_idx, pers), each (B, P) padded with -1 / -1 / -1.
    """
    B, n = values.shape
    dev = values.device
    g = _rank_keys(values)                                                   # unique int ranks
    idx = torch.arange(n, device=dev).expand(B, n)
    max_vals, _ = _sparse_table(g, "max")
    min_vals, min_args = _sparse_table(g, "min")
    K = max_vals.shape[1] - 1
    ar = torch.arange(B, device=dev)[:, None].expand(B, n)

    # previous greater: largest L < i with g[L] > g[i]   (pos = exclusive end of scanned block)
    pos = idx.clone()
    for k in range(K, -1, -1):
        start = pos - (1 << k)
        ok = start >= 0
        blk = max_vals[ar, k, start.clamp(min=0)]
        jump = ok & (blk < g)
        pos = torch.where(jump, start, pos)
    left = pos - 1                                                           # -1 if none
    # next greater: smallest R > i with g[R] > g[i]
    pos = idx + 1
    for k in range(K, -1, -1):
        ok = pos + (1 << k) <= n
        blk = max_vals[ar, k, pos.clamp(max=n - 1)]
        jump = ok & (blk < g)
        pos = torch.where(jump, pos + (1 << k), pos)
    right = pos                                                              # n if none

    interior = (idx > 0) & (idx < n - 1)
    is_death = interior & (g > torch.roll(g, 1, 1)) & (g > torch.roll(g, -1, 1))
    lmin = _range_argmin(min_vals, min_args, (left + 1).clamp(max=n - 1), (idx - 1).clamp(min=0))
    rmin = _range_argmin(min_vals, min_args, (idx + 1).clamp(max=n - 1), (right - 1).clamp(min=0))
    younger = torch.where(g.gather(1, lmin) > g.gather(1, rmin), lmin, rmin)
    birth = torch.where(is_death, younger, torch.full_like(idx, -1))
    death = torch.where(is_death, idx, torch.full_like(idx, -1))

    # essential class: global min paired with the last vertex processed (global max)
    gmin = (g == 0).float().argmax(dim=1)
    gmax = (g == n - 1).float().argmax(dim=1)

    count = is_death.sum(1)
    P = int(count.max().item()) + 1
    order = torch.argsort((~is_death).int(), dim=1, stable=True)[:, :P]      # deaths first, in index order
    birth = birth.gather(1, order)
    death = death.gather(1, order)
    slot = count.unsqueeze(1) == torch.arange(P, device=dev).unsqueeze(0)
    birth = torch.where(slot, gmin.unsqueeze(1).expand(-1, P), birth)
    death = torch.where(slot, gmax.unsqueeze(1).expand(-1, P), death)
    valid = birth >= 0
    vals = values.float()
    pers = torch.where(valid, vals.gather(1, death.clamp(min=0)) - vals.gather(1, birth.clamp(min=0)),
                       torch.full_like(vals[:, :1].expand(-1, P), -1.0))
    return birth, death, pers


def tibc_compress(values: torch.Tensor, num_keep: int = 144, return_indices: bool = False):
    """Topology-based image bytes compression (TIBC).

    Keeps the two endpoints plus the birth/death indices of the most persistent
    critical pairs of the byte signal until `num_keep` unique indices are
    collected (the `compress_tsc` rule), and returns the L2-normalised byte
    values at those indices, sorted by position and zero-padded to `num_keep`.
    """
    B, n = values.shape
    dev = values.device
    birth, death, pers = sublevel_persistence(values)
    P = birth.shape[1]
    # order pairs by persistence desc, ties by birth index asc (padding last)
    o1 = torch.argsort(birth, dim=1, stable=True)
    pers_s = pers.gather(1, o1)
    o2 = torch.argsort(-pers_s, dim=1, stable=True)
    order = o1.gather(1, o2)
    seq = torch.stack([birth.gather(1, order), death.gather(1, order)], dim=2).reshape(B, 2 * P)
    ends = torch.tensor([0, n - 1], device=dev).expand(B, 2)
    seq = torch.cat([ends, seq], dim=1)                                      # (B, 2 + 2P)
    valid = seq >= 0
    L = seq.shape[1]
    position = torch.arange(L, device=dev).expand(B, L)
    big = L + 1
    first = torch.full((B, n), big, dtype=torch.long, device=dev)
    first = first.scatter_reduce(1, seq.clamp(min=0), torch.where(valid, position, torch.full_like(position, big)),
                                 reduce="amin", include_self=True)
    is_first = valid & (first.gather(1, seq.clamp(min=0)) == position)
    rank = torch.cumsum(is_first.long(), dim=1)
    chosen = is_first & (rank <= num_keep)
    cand = torch.where(chosen, seq, torch.full_like(seq, n))
    idx = torch.sort(cand, dim=1).values[:, :num_keep]
    if idx.shape[1] < num_keep:
        idx = F.pad(idx, (0, num_keep - idx.shape[1]), value=n)
    pad = idx >= n
    idx = torch.where(pad, torch.full_like(idx, -1), idx)
    vals = torch.where(pad, torch.zeros_like(idx, dtype=torch.float),
                       values.float().gather(1, idx.clamp(min=0)))
    vals = F.normalize(vals, dim=1)
    return (vals, idx) if return_indices else vals


# --------------------------------------------------------------------------- #
# topology feature
# --------------------------------------------------------------------------- #
class TopologyFeature(nn.Module):
    """Conv1d(D -> 3, k=30, s=6) turns the 144 tokens into 20 points in R^3; their
    0-dim persistence (19 MST edge lengths) is lifted to a (num_tokens, D) feature."""

    def __init__(self, embed_dim: int, num_tokens: int = 144, point_dim: int = 3,
                 kernel_size: int = 30, stride: int = 6, hidden: Optional[int] = None):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(embed_dim, point_dim, kernel_size=kernel_size, stride=stride)
        n_points = (num_tokens - kernel_size) // stride + 1
        self.n_points = n_points
        self.increase = nn.Sequential(
            nn.Linear(n_points - 1, hidden or num_tokens, bias=False),
            nn.ReLU(),
            nn.Linear(hidden or num_tokens, num_tokens * embed_dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pts = self.conv(tokens.transpose(1, 2)).transpose(1, 2)             # (B, n_points, 3)
        pers = mst_persistence(pts.float())                                  # (B, n_points - 1)
        return self.increase(pers).reshape(tokens.shape[0], self.num_tokens, self.embed_dim)


# --------------------------------------------------------------------------- #
# ViT with topology cross-attention in the last block
# --------------------------------------------------------------------------- #
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TopoAttention(nn.Module):
    """Self-attention whose keys/values can come from tokens + topology feature."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(x).reshape(B, N, h, C // h).transpose(1, 2)
        kv = self.kv(context).reshape(B, N, 2, h, C // h).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(enabled=False):
            q, k, v = q.float(), kv[0].float(), kv[1].float()
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TopoBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., use_topology=False):
        super().__init__()
        self.use_topology = use_topology
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = TopoAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, topo=None):
        xn = self.norm1(x)
        context = xn + self.norm1(topo) if (self.use_topology and topo is not None) else xn
        x = x + self.drop_path(self.attn(xn, context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransFacePPViT(nn.Module):
    """Byte-level ViT of TransFace++ producing a `num_classes`-d face embedding.

    forward(byte_seq):  byte_seq is (B, num_bytes) int64 in [0, 255].
      train: (embedding, patch_entropy)             [norm_output: (embedding, norm, patch_entropy)]
      eval:  embedding                              [norm_output: (embedding, norm)]
    """

    def __init__(self, num_bytes: int, num_classes: int = 512, embed_dim: int = 512, depth: int = 12,
                 num_heads: int = 8, mlp_ratio: float = 4., qkv_bias: bool = False, qk_scale=None,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.,
                 mask_ratio: float = 0.0, using_checkpoint: bool = False, num_tokens: int = 144,
                 use_topology: bool = True, tibc_prob: float = 0.3,
                 fp16: bool = False, norm_output: bool = False):
        super().__init__()
        self.num_bytes = num_bytes
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint
        self.tibc_prob = tibc_prob
        self.fp16 = fp16
        self.norm_output = norm_output

        self.tokenizer = ByteTokenizer(num_bytes, embed_dim, num_tokens)
        self.topology = TopologyFeature(embed_dim, num_tokens) if use_topology else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TopoBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                      dpr[i], use_topology=(use_topology and i == depth - 1))
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.senet = nn.Sequential(
            nn.Linear(embed_dim * num_tokens, num_tokens, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_tokens, num_tokens, bias=False),
            nn.Sigmoid())
        self.feature = nn.Sequential(
            nn.Linear(embed_dim * num_tokens, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, eps=2e-5),
            nn.Linear(embed_dim, num_classes, bias=False),
            nn.BatchNorm1d(num_classes, eps=2e-5))
        torch.nn.init.normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'mask_token'}

    def random_masking(self, x, topo):
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep].unsqueeze(-1).repeat(1, 1, D)
        x = torch.gather(x, 1, ids_keep)
        topo = torch.gather(topo, 1, ids_keep) if topo is not None else None
        return x, topo, ids_restore

    def _tibc(self, byte_seq: torch.Tensor) -> Optional[torch.Tensor]:
        B = byte_seq.shape[0]
        sel = torch.rand(B, device=byte_seq.device) < self.tibc_prob
        if not sel.any():
            return None
        add = torch.zeros(B, self.num_tokens, device=byte_seq.device)
        chunks = torch.nonzero(sel).flatten().split(32)
        for c in chunks:
            add[c] = tibc_compress(byte_seq[c].float(), self.num_tokens)
        return add

    def forward_features(self, byte_seq: torch.Tensor):
        B = byte_seq.shape[0]
        x = self.tokenizer(byte_seq)
        topo = self.topology(x) if self.topology is not None else None
        if self.training and self.tibc_prob > 0:
            add = self._tibc(byte_seq)
            if add is not None:
                x = x + add.unsqueeze(-1).to(x.dtype)
        x = self.pos_drop(x + self.pos_embed)
        ids_restore = None
        if self.training and self.mask_ratio > 0:
            x, topo, ids_restore = self.random_masking(x, topo)
        for blk in self.blocks:
            if self.using_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(blk, x, topo, use_reentrant=True)
            else:
                x = blk(x, topo)
        x = self.norm(x.float())
        if ids_restore is not None:
            mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - x.shape[1], 1)
            x = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        weight = self.senet(x.reshape(B, -1))                                # (B, num_tokens) in (0, 1)
        x = x * weight.unsqueeze(-1)
        return x

    def forward(self, byte_seq: torch.Tensor):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.forward_features(byte_seq)
        x = x.float()
        patch_entropy = x.std(dim=2)                                         # (B, num_tokens)
        emb = self.feature(x.reshape(x.shape[0], -1))
        outputs = (emb,)
        if self.norm_output:
            norm = torch.norm(emb, 2, 1, True)
            outputs = (torch.div(emb, norm), norm)
        if self.training:
            return (*outputs, patch_entropy)
        return outputs[0] if len(outputs) == 1 else outputs


def transface_pp_vit_s(num_bytes, **kwargs):
    return TransFacePPViT(num_bytes, embed_dim=512, depth=12, num_heads=8, drop_path_rate=0.05,
                          mask_ratio=0.0, **kwargs)


def transface_pp_vit_b(num_bytes, **kwargs):
    return TransFacePPViT(num_bytes, embed_dim=512, depth=24, num_heads=8, drop_path_rate=0.05,
                          mask_ratio=0.05, using_checkpoint=True, **kwargs)
