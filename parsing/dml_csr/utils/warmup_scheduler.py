#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author     :   Qingping Zheng
@Contact    :   qingpingzheng2014@gmail.com
@File       :   warmup_scheduler.py
@Time       :   10/01/21 00:00 PM
@Desc       :   
@License    :   Licensed under the Apache License, Version 2.0 (the "License"); 
@Copyright  :   Copyright 2022 The Authors. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up learning rate with cosine annealing in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    """

    def __init__(self, optimizer, total_epoch, eta_min=0, warmup_epoch=10, last_epoch=-1):
        self.total_epoch = total_epoch
        self.eta_min = eta_min
        self.warmup_epoch = warmup_epoch
        super(GradualWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_epoch:
            return [self.eta_min + self.last_epoch*(base_lr - self.eta_min)/self.warmup_epoch for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr-self.eta_min)*(1+math.cos(math.pi*(self.last_epoch-self.warmup_epoch)/(self.total_epoch-self.warmup_epoch))) / 2 for base_lr in self.base_lrs]


class SGDRScheduler(_LRScheduler):
    """ Consine annealing with warm up and restarts.
    Proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`.
    """
    def __init__(self, optimizer, total_epoch=150, start_cyclical=100, cyclical_base_lr=7e-4, cyclical_epoch=10, eta_min=0, warmup_epoch=10, last_epoch=-1):
        self.total_epoch = total_epoch
        self.start_cyclical = start_cyclical
        self.cyclical_epoch = cyclical_epoch
        self.cyclical_base_lr = cyclical_base_lr
        self.eta_min = eta_min
        self.warmup_epoch = warmup_epoch
        super(SGDRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [self.eta_min + self.last_epoch*(base_lr - self.eta_min)/self.warmup_epoch for base_lr in self.base_lrs]
        elif self.last_epoch < self.start_cyclical:
            return [self.eta_min + (base_lr-self.eta_min)*(1+math.cos(math.pi*(self.last_epoch-self.warmup_epoch)/(self.start_cyclical-self.warmup_epoch))) / 2 for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (self.cyclical_base_lr-self.eta_min)*(1+math.cos(math.pi* ((self.last_epoch-self.start_cyclical)% self.cyclical_epoch)/self.cyclical_epoch)) / 2 for base_lr in self.base_lrs]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=7e-3, momentum=0.9, weight_decay=5e-4)
    scheduler_warmup = SGDRScheduler(optimizer, total_epoch=150, eta_min=7e-5, warmup_epoch=10, start_cyclical=100, cyclical_base_lr=3.5e-3, cyclical_epoch=10)
    lr = []
    for epoch in range(0,150):
        scheduler_warmup.step(epoch)
        lr.append(scheduler_warmup.get_lr())
    plt.style.use('ggplot')
    plt.plot(list(range(0,150)), lr)
    plt.show()
