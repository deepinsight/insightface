"""
Author: {Yang Xiao, Xiang An, XuHan Zhu} in DeepGlint,
Partial FC: Training 10 Million Identities on a Single Machine
See the original paper:
https://arxiv.org/abs/2010.05222
"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import backbones
from config import config as cfg
from dataset import MXFaceDataset, DataLoaderX
from partial_classifier import DistSampleClassifier
from sgd import SGD

torch.backends.cudnn.benchmark = True


class MarginSoftmax(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(MarginSoftmax, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0],
                            cosine.size()[1],
                            device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# .......
def main(local_rank):
    dist.init_process_group(backend='nccl', init_method='env://')
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size()
    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    train_loader = DataLoaderX(local_rank=local_rank,
                               dataset=trainset,
                               batch_size=cfg.batch_size,
                               sampler=train_sampler,
                               num_workers=0,
                               pin_memory=True,
                               drop_last=False)

    backbone = backbones.iresnet100(False).to(local_rank)
    backbone.train()

    # Broadcast init parameters
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    # DDP
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=[cfg.local_rank])
    backbone.train()

    # Memory classifer
    dist_sample_classifer = DistSampleClassifier(
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=cfg.world_size)

    # Margin softmax
    margin_softmax = MarginSoftmax(s=64.0, m=0.4)

    # Optimizer for backbone and classifer
    optimizer = SGD([{
        'params': backbone.parameters()
    }, {
        'params': dist_sample_classifer.parameters()
    }],
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        rescale=cfg.world_size)

    # Lr scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=cfg.lr_func)
    n_epochs = cfg.num_epoch
    start_epoch = 0

    if local_rank == 0:
        writer = SummaryWriter(log_dir='logs/shows')

    #
    total_step = int(len(trainset) / cfg.batch_size / dist.get_world_size() * cfg.num_epoch)
    if dist.get_rank() == 0:
        print("Total Step is: %d" % total_step)

    losses = AverageMeter()
    global_step = 0
    train_start = time.time()
    for epoch in range(start_epoch, n_epochs):
        train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            total_label, norm_weight = dist_sample_classifer.prepare(
                label, optimizer)
            features = F.normalize(backbone(img))

            # Features all-gather
            total_features = torch.zeros(features.size()[0] * cfg.world_size,
                                         cfg.embedding_size,
                                         device=local_rank)
            dist.all_gather(list(total_features.chunk(cfg.world_size, dim=0)),
                            features.data)
            total_features.requires_grad = True

            # Calculate logits
            logits = dist_sample_classifer(total_features, norm_weight)
            logits = margin_softmax(logits, total_label)

            with torch.no_grad():
                max_fc = torch.max(logits, dim=1, keepdim=True)[0]
                dist.all_reduce(max_fc, dist.ReduceOp.MAX)

                # Calculate exp(logits) and all-reduce
                logits_exp = torch.exp(logits - max_fc)
                logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
                dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

                # Calculate prob
                logits_exp.div_(logits_sum_exp)

                # Get one-hot
                grad = logits_exp
                index = torch.where(total_label != -1)[0]
                one_hot = torch.zeros(index.size()[0],
                                      grad.size()[1],
                                      device=grad.device)
                one_hot.scatter_(1, total_label[index, None], 1)

                # Calculate loss
                loss = torch.zeros(grad.size()[0], 1, device=grad.device)
                loss[index] = grad[index].gather(1, total_label[index, None])
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

                # Calculate grad
                grad[index] -= one_hot
                grad.div_(features.size()[0])

            logits.backward(grad)
            if total_features.grad is not None:
                total_features.grad.detach_()
            x_grad = torch.zeros_like(features)

            # Feature gradient all-reduce
            dist.reduce_scatter(
                x_grad, list(total_features.grad.chunk(cfg.world_size, dim=0)))
            x_grad.mul_(cfg.world_size)
            # Backward backbone
            features.backward(x_grad)
            optimizer.step()

            # Update classifer
            dist_sample_classifer.update()
            optimizer.zero_grad()
            losses.update(loss_v, 1)
            if cfg.local_rank == 0 and step % 50 == 0:
                time_now = (time.time() - train_start) / 3600
                time_total = time_now / ((global_step + 1) / total_step)
                time_for_end = time_total - time_now
                writer.add_scalar('time_for_end', time_for_end, global_step)
                writer.add_scalar('loss', loss_v, global_step)
                print("Speed %d samples/sec   Loss %.4f   Epoch: %d   Global Step: %d   Required: %1.f hours" %
                      (
                          (cfg.batch_size * global_step / (time.time() - train_start) * cfg.world_size),
                          losses.avg,
                          epoch,
                          global_step,
                          time_for_end
                      ))
                losses.reset()

            global_step += 1
        scheduler.step()
        if dist.get_rank() == 0:
            import os
            if not os.path.exists(cfg.output):
                os.makedirs(cfg.output)
            torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(epoch) + 'backbone.pth'))
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args = parser.parse_args()
    main(args.local_rank)
