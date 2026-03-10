"""
Sub-center ArcFace variant of PartialFC_V2.

Ref: Sub-center ArcFace: Boosting Face Recognition by Enhancing Intra-class Compactness
     https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf

Instead of a single weight vector per class, K sub-centers are maintained.
The logit for each class = max over K sub-center similarities.
This makes training more robust to noisy labels and intra-class variance.
"""

import math
from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize

from partial_fc_v2 import DistCrossEntropy, AllGather


class PartialFC_V2_SubCenter(torch.nn.Module):
    """
    Sub-center variant of PartialFC_V2.
    
    Each class has K sub-centers. During forward:
    1. Compute cosine similarity with all sub-centers (num_classes * K)
    2. Reshape to (batch, num_classes, K) and take max over K
    3. Apply the margin loss on the resulting (batch, num_classes) logits
    4. Compute cross-entropy loss
    
    Parameters:
    -----------
    margin_loss: Callable
        Margin-based loss function (e.g., CombinedMarginLoss)
    embedding_size: int
        Dimension of face embeddings
    num_classes: int
        Total number of identity classes
    num_subcenters: int
        Number of sub-centers per class (K). Default: 3
    sample_rate: float
        Negative class sampling rate for PartialFC. Default: 1.0
    fp16: bool
        Use mixed precision for logit computation
    """
    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        num_subcenters: int = 3,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        super().__init__()
        assert distributed.is_initialized(), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.num_subcenters = num_subcenters
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True

        # K sub-centers per class: shape = (num_local * K, embedding_size)
        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.num_local * num_subcenters, embedding_size))
        )

        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise ValueError("margin_loss must be callable")

    def sample(self, labels, index_positive):
        """
        Sample a subset of classes for PartialFC.
        Adapted for sub-center: each sampled class has K weight rows.
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).cuda()
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local]).cuda()
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1].cuda()
                index = index.sort()[0].cuda()
            else:
                index = positive
            self.weight_index_classes = index

            # Map class indices to sub-center weight indices
            # For class i, sub-center weights are at rows [i*K, i*K+1, ..., i*K+(K-1)]
            K = self.num_subcenters
            weight_indices = []
            for k in range(K):
                weight_indices.append(index * K + k)
            self.weight_index = torch.cat(weight_indices).sort()[0]

            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        return self.weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        K = self.num_subcenters
        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
            num_classes_active = self.num_sample
        else:
            weight = self.weight
            num_classes_active = self.num_local

        with torch.amp.autocast('cuda', enabled=self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight = normalize(weight)
            # logits shape: (total_batch, num_classes_active * K)
            logits = linear(norm_embeddings, norm_weight)

        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        # Reshape to (batch, num_classes_active, K) and take max over sub-centers
        logits = logits.view(logits.size(0), num_classes_active, K)
        logits = logits.max(dim=2)[0]  # (batch, num_classes_active)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss

    def get_dominant_subcenters(self):
        """
        After training, extract the dominant sub-center (highest norm) for each class.
        Returns a weight matrix of shape (num_local, embedding_size) for standard inference.
        """
        K = self.num_subcenters
        with torch.no_grad():
            w = self.weight.view(self.num_local, K, self.embedding_size)
            norms = w.norm(dim=2)  # (num_local, K)
            dominant_idx = norms.argmax(dim=1)  # (num_local,)
            dominant_weight = w[torch.arange(self.num_local), dominant_idx]
        return dominant_weight
