from default import config
import mxnet as mx


def get_scheduler():
    step = [int(x) for x in config.lr_steps.split(',')]
    backbone_lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
        step=step, factor=0.1, base_lr=config.backbone_lr)
    memory_bank_lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
        step=step, factor=0.1, base_lr=config.memory_bank_lr)

    return backbone_lr_scheduler, memory_bank_lr_scheduler
