"""MambaVision backbone for 112x112 face recognition.

Port of NVlabs/MambaVision (Hatamizadeh & Kautz, "MambaVision: A Hybrid
Mamba-Transformer Vision Backbone", 2024) -- https://github.com/NVlabs/MambaVision
(mambavision/models/mamba_vision.py, Apache-2.0 / NVIDIA Source Code License).

Architecture (faithful to the original):
    PatchEmbed (stride 4)
    level 0: ConvBlock x depths[0]                 stride 4   (28x28 @112)
    level 1: ConvBlock x depths[1]                 stride 8   (14x14)
    level 2: [MambaVisionMixer..., Attention...]   stride 16  (7x7)
    level 3: [MambaVisionMixer..., Attention...]   stride 32  (4x4)
    BN -> GAP -> Linear -> BN  (512-d embedding)

Adaptations for faces: the default window sizes are [7, 7, 7, 4] so the 7x7 /
4x4 grids at 112x112 are processed as one window without zero padding; the
classifier is replaced by an embedding head with the same `fp16` /
`norm_output` contract as the other arcface_torch backbones.

The selective scan uses the fused CUDA kernel from `mamba_ssm` when it is
installed and falls back to a pure-PyTorch implementation of the same
recurrence otherwise (the SSM only runs on 49 / 16 tokens here, so the
fallback is cheap).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp


# --------------------------------------------------------------------------- #
# selective scan
# --------------------------------------------------------------------------- #
def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """Reference S6 selective scan with the `mamba_ssm.selective_scan_fn` signature.

    u, delta: (B, D, L)   A: (D, N)   B, C: (B, N, L)   D, delta_bias: (D,)
    h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t ;  y_t = C_t . h_t + D * u_t
    """
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    B, C = B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    n_state = A.shape[1]
    deltaA = torch.exp(delta.unsqueeze(-1) * A.float()[None, :, None, :])       # (B, D, L, N)
    deltaBu = delta.unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1) * u.unsqueeze(-1)
    Ct = C.transpose(1, 2)                                                       # (B, L, N)
    h = u.new_zeros(u.shape[0], u.shape[1], n_state)
    ys = []
    for t in range(u.shape[2]):
        h = deltaA[:, :, t] * h + deltaBu[:, :, t]
        ys.append((h * Ct[:, t].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    y = y.to(dtype_in)
    return (y, h) if return_last_state else y


try:  # fused CUDA kernel (optional dependency)
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_cuda
except ImportError:  # pragma: no cover - depends on environment
    _selective_scan_cuda = None

selective_scan_fn = _selective_scan_cuda if _selective_scan_cuda is not None else selective_scan_ref


# --------------------------------------------------------------------------- #
# building blocks (as in NVlabs/MambaVision)
# --------------------------------------------------------------------------- #
def window_partition(x, window_size):
    """(B, C, H, W) -> (num_windows*B, window_size*window_size, C)"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window_size*window_size, C) -> (B, C, H, W)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)


class Downsample(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False))

    def forward(self, x):
        return self.reduction(x)


class PatchEmbed(nn.Module):
    """Two stride-2 convs: 3 -> in_dim -> dim, overall stride 4."""

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_down(x)


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale is not None and type(layer_scale) in (int, float)
        if self.layer_scale:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        inp = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        return inp + self.drop_path(x)


class MambaVisionMixer(nn.Module):
    """MambaVision token mixer: SSM on half of the channels, conv+SiLU on the other half, concat.

    Defaults (d_state=8, d_conv=3, expand=1) are the values MambaVision uses in every Block.
    """

    def __init__(self, d_model, d_state=8, d_conv=3, expand=1, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        half = self.d_inner // 2
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        self.x_proj = nn.Linear(half, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, half, bias=True)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(half) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(half, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(half))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.conv1d_x = nn.Conv1d(half, half, bias=conv_bias // 2, kernel_size=d_conv, groups=half)
        self.conv1d_z = nn.Conv1d(half, half, bias=conv_bias // 2, kernel_size=d_conv, groups=half)

    def forward(self, hidden_states):
        """(B, L, D) -> (B, L, D)"""
        _, seqlen, _ = hidden_states.shape
        half = self.d_inner // 2
        xz = self.in_proj(hidden_states).transpose(1, 2)                          # (B, d_inner, L)
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(x, self.conv1d_x.weight, self.conv1d_x.bias, padding='same', groups=half))
        z = F.silu(F.conv1d(z, self.conv1d_z.weight, self.conv1d_z.bias, padding='same', groups=half))
        x_dbl = self.x_proj(x.transpose(1, 2).reshape(-1, half))                   # (B*L, rank+2N)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).view(-1, seqlen, half).transpose(1, 2)               # (B, half, L)
        B = B.view(-1, seqlen, self.d_state).transpose(1, 2).contiguous()          # (B, N, L)
        C = C.view(-1, seqlen, self.d_state).transpose(1, 2).contiguous()
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None,
                              delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
                              return_last_state=False)
        y = torch.cat([y, z], dim=1).transpose(1, 2)                               # (B, L, d_inner)
        return self.out_proj(y)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, counter, transformer_blocks, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_scale,
                                   attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in (int, float)
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """One resolution level: ConvBlocks (levels 0-1) or windowed mixer/attention Blocks (levels 2-3)."""

    def __init__(self, dim, depth, num_heads, window_size, conv=False, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., layer_scale=None, layer_scale_conv=None, transformer_blocks=()):
        super().__init__()
        self.conv = conv
        dp = lambda i: drop_path[i] if isinstance(drop_path, list) else drop_path
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim, drop_path=dp(i), layer_scale=layer_scale_conv)
                                         for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim, counter=i, transformer_blocks=transformer_blocks,
                                               num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                               drop_path=dp(i), layer_scale=layer_scale)
                                         for i in range(depth)])
            self.transformer_block = True
        self.downsample = None if not downsample else Downsample(dim=dim)
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
            Hp, Wp = x.shape[-2:]
            x = window_partition(x, self.window_size)
        for blk in self.blocks:
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaVision(nn.Module):
    """Hierarchical MambaVision backbone producing a `num_features`-d face embedding."""

    def __init__(self, dim, in_dim, depths, window_size, mlp_ratio, num_heads,
                 drop_path_rate=0.2, in_chans=3, num_features=512, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., layer_scale=None, layer_scale_conv=None,
                 fp16=False, norm_output=False, **kwargs):
        super().__init__()
        self.fp16 = fp16
        self.norm_output = norm_output
        out_dim = int(dim * 2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            d = depths[i]
            transformer_blocks = list(range(d // 2 + 1, d)) if d % 2 != 0 else list(range(d // 2, d))
            self.levels.append(MambaVisionLayer(
                dim=int(dim * 2 ** i), depth=d, num_heads=num_heads[i], window_size=window_size[i],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, conv=(i in (0, 1)),
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=(i < 3),
                layer_scale=layer_scale, layer_scale_conv=layer_scale_conv,
                transformer_blocks=transformer_blocks))
        self.norm = nn.BatchNorm2d(out_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # embedding head (replaces the ImageNet classifier)
        self.fc = nn.Linear(out_dim, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'A_log', 'D'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.forward_features(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        if self.norm_output:
            norm = torch.norm(x, 2, 1, True)
            return torch.div(x, norm), norm
        return x


# --------------------------------------------------------------------------- #
# model zoo (dims/depths/heads as in NVlabs; window sizes adapted to 112x112)
# --------------------------------------------------------------------------- #
_FACE_WINDOWS = [7, 7, 7, 4]


def mambavision_t(**kwargs):
    return MambaVision(dim=80, in_dim=32, depths=[1, 3, 8, 4], num_heads=[2, 4, 8, 16],
                       window_size=_FACE_WINDOWS, mlp_ratio=4, drop_path_rate=0.2, **kwargs)


def mambavision_s(**kwargs):
    return MambaVision(dim=96, in_dim=64, depths=[3, 3, 7, 5], num_heads=[2, 4, 8, 16],
                       window_size=_FACE_WINDOWS, mlp_ratio=4, drop_path_rate=0.2, **kwargs)


def mambavision_b(**kwargs):
    return MambaVision(dim=128, in_dim=64, depths=[3, 3, 10, 5], num_heads=[2, 4, 8, 16],
                       window_size=_FACE_WINDOWS, mlp_ratio=4, drop_path_rate=0.3, **kwargs)


def mambavision_l(**kwargs):
    return MambaVision(dim=196, in_dim=64, depths=[3, 3, 10, 5], num_heads=[4, 8, 16, 32],
                       window_size=_FACE_WINDOWS, mlp_ratio=4, drop_path_rate=0.3,
                       layer_scale=1e-5, **kwargs)
