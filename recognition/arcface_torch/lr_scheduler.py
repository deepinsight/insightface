from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import SGD
import torch
import warnings

class PolynomialLRWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        self.total_iters = total_iters
        self.power = power
        self.warmup_iters = warmup_iters


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch <= self.warmup_iters:
            return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        else:        
            l = self.last_epoch
            w = self.warmup_iters
            t = self.total_iters
            decay_factor = ((1.0 - (l - w) / (t - w)) / (1.0 - (l - 1 - w) / (t - w))) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):

        if self.last_epoch <= self.warmup_iters:
            return [
                base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        else:
            return [
                (
                    base_lr * (1.0 - (min(self.total_iters, self.last_epoch) - self.warmup_iters) / (self.total_iters - self.warmup_iters)) ** self.power
                )
                for base_lr in self.base_lrs
            ]

    
if __name__ == "__main__":

    class TestModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)
        
        def forward(self, x):
            return self.linear(x)

    test_module = TestModule()
    test_module_pfc = TestModule()
    lr_pfc_weight = 1 / 3
    base_lr = 10
    total_steps = 1000
    
    sgd = SGD([
        {"params": test_module.parameters(), "lr": base_lr},
        {"params": test_module_pfc.parameters(), "lr": base_lr * lr_pfc_weight}
        ], base_lr)

    scheduler = PolynomialLRWarmup(sgd, total_steps//10, total_steps, power=2)

    x = []
    y = []
    y_pfc = []
    for i in range(total_steps):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        lr_pfc = scheduler.get_last_lr()[1]
        x.append(i)
        y.append(lr)
        y_pfc.append(lr_pfc)

    import matplotlib.pyplot as plt
    fontsize=15
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, linestyle='-', linewidth=2, )
    plt.plot(x, y_pfc, linestyle='-', linewidth=2, )
    plt.xlabel('Iterations')     # x_label
    plt.ylabel("Lr")             # y_label
    plt.savefig("tmp.png", dpi=600, bbox_inches='tight')
