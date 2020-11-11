"""
Author: {Yang Xiao, Xiang An, XuHan Zhu} in DeepGlint,
Partial FC: Training 10 Million Identities on a Single Machine
See the original paper:
https://arxiv.org/abs/2010.05222
"""

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter

from config import config as cfg


class DistSampleClassifier(Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    @torch.no_grad()
    def __init__(self, rank, local_rank, world_size):
        super(DistSampleClassifier, self).__init__()
        self.sample_rate = cfg.sample_rate
        self.num_local = cfg.num_classes // world_size + int(
            rank < cfg.num_classes % world_size)
        self.class_start = cfg.num_classes // world_size * rank + min(
            rank, cfg.num_classes % world_size)
        self.num_sample = int(self.sample_rate * self.num_local)
        self.local_rank = local_rank
        self.world_size = world_size

        self.weight = torch.empty(size=(self.num_local, cfg.embedding_size),
                                  device=local_rank)
        self.weight_mom = torch.zeros_like(self.weight)
        self.stream = torch.cuda.Stream(local_rank)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = Parameter(torch.empty((0, 0)).cuda(local_rank))


    @torch.no_grad()
    def sample(self, total_label):
        P = (self.class_start <=
             total_label) & (total_label < self.class_start + self.num_local)
        total_label[~P] = -1
        total_label[P] -= self.class_start
        if int(self.sample_rate) != 1:
            positive = torch.unique(total_label[P], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(self.num_local,device=cfg.local_rank)
                perm[positive] = 2.0
                index = torch.topk(perm,k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.index = index
            total_label[P] = torch.searchsorted(index, total_label[P])
            self.sub_weight = Parameter(self.weight[index])
            self.sub_weight_mom = self.weight_mom[index]

    def forward(self, total_features, norm_weight):
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = F.linear(total_features, norm_weight)
        return logits

    @torch.no_grad()
    def update(self, ):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        with torch.cuda.stream(self.stream):
            total_label = torch.zeros(label.size()[0] * self.world_size,
                                      device=self.local_rank,
                                      dtype=torch.long)
            dist.all_gather(list(total_label.chunk(self.world_size, dim=0)),
                            label)
            self.sample(total_label)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[
                self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = F.normalize(self.sub_weight)
            return total_label, norm_weight
