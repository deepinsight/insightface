import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 base_lr,
                 max_steps,
                 warmup_steps,
                 last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, last_epoch, False)

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        #_lr = max(self.base_lr * alpha, self.warmup_lr_init)
        _lr = self.base_lr * alpha
        return [_lr for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1 - float(self.last_epoch - self.warmup_steps) /
                float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]

class StepScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 base_lr,
                 lr_steps,
                 warmup_steps,
                 last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.lr_steps = lr_steps
        self.warmup_steps: int = warmup_steps
        super(StepScheduler, self).__init__(optimizer, last_epoch, False)

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        #_lr = max(self.base_lr * alpha, self.warmup_lr_init)
        _lr = self.base_lr * alpha
        return [_lr for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = 0.1 ** len([m for m in self.lr_steps if m <= self.last_epoch])
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]



def get_scheduler(opt, cfg):
    if cfg.lr_func is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt, lr_lambda=cfg.lr_func)
    else:
        #total_batch_size = cfg.batch_size * cfg.world_size
        #warmup_steps = cfg.num_images // total_batch_size * cfg.warmup_epochs
        #total_steps = cfg.num_images // total_batch_size * cfg.num_epochs

        if cfg.lr_steps is None:
            scheduler = PolyScheduler(
                optimizer=opt,
                base_lr=cfg.lr,
                max_steps=cfg.total_steps,
                warmup_steps=cfg.warmup_steps,
            )
        else:
            scheduler = StepScheduler(
                optimizer=opt,
                base_lr=cfg.lr,
                lr_steps=cfg.lr_steps,
                warmup_steps=cfg.warmup_steps,
            )

    return scheduler

