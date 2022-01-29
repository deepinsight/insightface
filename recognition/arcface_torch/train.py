import argparse
import logging
import os

import torch
from torch import distributed
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from torch.utils.data import DataLoader
from lr_scheduler import PolyScheduler
from losses import CosFace, ArcFace
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali)
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank])
    backbone.train()

    if cfg.loss == "arcface":
        margin_loss = ArcFace()
    elif cfg.loss == "cosface":
        margin_loss = CosFace()
    else:
        raise

    module_partial_fc = PartialFC(
        margin_loss,
        cfg.embedding_size, 
        cfg.num_classes, 
        cfg.sample_rate, 
        cfg.fp16
    )
    module_partial_fc.train().cuda()

    # TODO the params of partial fc must be last in the params list
    opt = torch.optim.SGD(
        params=[
            {"params": backbone.parameters(), },
            {"params": module_partial_fc.parameters(), },
        ],
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay
    )
    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // total_batch_size * cfg.num_epoch
    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step
    )

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    start_epoch = 0
    global_step = 0
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 200:
                    callback_verification(global_step, backbone)
        
        path_pfc = os.path.join(cfg.output, "softmax_fc_gpu_{}.pt".format(rank))
        torch.save(module_partial_fc.state_dict(), path_pfc)
        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)
        
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
