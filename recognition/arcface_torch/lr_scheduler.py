from typing import *
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler


def get_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'PolyScheduler':
        return PolyScheduler(
            optimizer=optimizer, base_lr=cfg.lr,
            max_steps=cfg.total_step,
            warmup_steps=cfg.warmup_step,
            last_epoch=-1
        )
    if cfg.lr_scheduler == 'MultiStepScheduler':
        cfg.milestone_steps = [
            x * cfg.steps_per_epoch
            for x in cfg.decay_epoch
        ]
        return MultiStepScheduler(
            optimizer=optimizer, base_lr=cfg.lr,
            milestones=cfg.milestone_steps,
            gamma=cfg.gamma,
            warmup_steps=cfg.warmup_step,
            last_epoch=-1
        )
    assert False, f'NotImplemented {cfg.lr_scheduler}'


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]


class MultiStepScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr,
                 milestones: Sequence[int], gamma: float, warmup_steps: int,
                 last_epoch=-1):
        self.base_lr = base_lr
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_steps: int = warmup_steps
        self.warmup_lr_init = 0.0001
        super(MultiStepScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = self.gamma ** bisect_right(self.milestones, self.last_epoch)
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]
