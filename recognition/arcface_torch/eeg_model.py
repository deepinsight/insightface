import math
import torch
from torch import nn
import torch.nn.functional as F

class TemporalSpatialEEGEncoder(nn.Module):
    """Custom EEG encoder. Input [B, channels, time], output [B, embedding_dim]."""
    def __init__(self, channels=8, embedding_dim=32, width=16, dropout=0.25):
        super().__init__()
        # Multi-scale temporal filters: all weights initialized from scratch.
        self.temporal_short = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(width), nn.GELU())
        self.temporal_mid = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(width), nn.GELU())
        self.temporal_long = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=(1, 49), padding=(0, 24), bias=False),
            nn.BatchNorm2d(width), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width * 2, kernel_size=(channels, 1), bias=False),
            nn.BatchNorm2d(width * 2), nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Conv2d(width * 2, width * 2, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(width * 2), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim, affine=False))
        self.channels = channels
        self.embedding_dim = embedding_dim

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"Expected [B,{self.channels},T], got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        features = torch.cat([
            self.temporal_short(x),
            self.temporal_mid(x),
            self.temporal_long(x)
        ], dim=1)
        z = self.fuse(features)
        return F.normalize(z, p=2, dim=1)

class ArcFaceHead(nn.Module):
    """Standard small-class ArcFace head, using normalized embeddings and weights."""
    def __init__(self, embedding_dim, num_classes, scale=32.0, margin=0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale
