"""TransFace ViT (Dan et al., ICCV 2023): ViT + SE patch re-weighting.

As in the official implementation (TransFace/backbones/vit.py), an SE module on
the flattened patch features produces per-patch gates; the gated features go to
the embedding head. In training the model additionally returns
    patch_weight  (B, P)  softmax of the SE gates   -> picks the dominant patches for DPAP
    patch_entropy (B, P)  std of each patch feature -> EHSM sample weights
"""
import torch
import torch.nn as nn

from .vit import VisionTransformer


class TransFaceViT(VisionTransformer):
    """forward(x):
        train: (embedding, patch_weight, patch_entropy)      [norm_output: (embedding, norm, patch_weight, patch_entropy)]
        eval:  embedding                                     [norm_output: (embedding, norm)]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        P, D = self.num_patches, self.embed_dim
        self.senet = nn.Sequential(
            nn.Linear(P * D, P, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(P, P, bias=False),
            nn.Sigmoid())
        self._patch_weight = None
        self._patch_entropy = None

    def forward_features(self, x):
        feats = super().forward_features(x)                  # (B, P*D), fp32 after the final norm
        B = feats.shape[0]
        gate = self.senet(feats)                              # (B, P) in (0, 1)
        tokens = feats.reshape(B, self.num_patches, self.embed_dim) * gate.unsqueeze(-1)
        self._patch_weight = gate.softmax(dim=1)
        self._patch_entropy = tokens.std(dim=2)
        return tokens.reshape(B, -1)

    def forward(self, x):
        out = super().forward(x)
        if self.training:
            out = out if isinstance(out, tuple) else (out,)
            return (*out, self._patch_weight, self._patch_entropy)
        return out
