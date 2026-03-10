"""
Recover sub-center FC weights from a trained backbone.

If train_v2_subcenter.py was run without saving FC weights (e.g., save_all_states
was False and the old code didn't save subcenter_fc_gpu_*.pt), this script recovers
them by freezing the backbone and training only the sub-center FC head for a few epochs.

With a well-trained backbone, the sub-center FC converges quickly (~2-3 epochs).

Usage (distributed, same as training):
    torchrun --nproc_per_node=3 recover_subcenter_fc.py configs/subcenter_merged_ms1m_glint_r100 \
        --backbone /output/subcenter_merged_ms1m_glint_r100/model.pt \
        --epochs 3
"""
import argparse
import logging
import os
import warnings

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2_subcenter import PartialFC_V2_SubCenter
from torch import distributed
from torch.utils.data import DataLoader
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging

assert torch.__version__ >= "1.12.0"

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    # Load backbone and FREEZE it
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).cuda()

    ckpt = torch.load(args.backbone, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict_backbone' in ckpt:
        ckpt = ckpt['state_dict_backbone']
    new_state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    backbone.load_state_dict(new_state)

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank],
        bucket_cap_mb=16, find_unused_parameters=False
    )

    # Freeze backbone — we only train the FC head
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    logging.info("Backbone loaded and FROZEN from %s", args.backbone)

    # Data
    train_loader = get_dataloader(
        cfg.rec, local_rank, cfg.batch_size, cfg.dali, cfg.dali_aug, cfg.seed, cfg.num_workers
    )

    # Loss and sub-center FC head
    margin_loss = CombinedMarginLoss(
        64, cfg.margin_list[0], cfg.margin_list[1], cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    num_subcenters = getattr(cfg, 'num_subcenters', 3)
    logging.info(f"Sub-center FC recovery: K={num_subcenters}")

    module_partial_fc = PartialFC_V2_SubCenter(
        margin_loss, cfg.embedding_size, cfg.num_classes,
        num_subcenters=num_subcenters,
        sample_rate=cfg.sample_rate, fp16=False
    )
    module_partial_fc.train().cuda()

    # Optimizer — only FC parameters
    opt = torch.optim.SGD(
        params=[{"params": module_partial_fc.parameters()}],
        lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
    )

    num_epochs = args.epochs
    total_batch_size = cfg.batch_size * world_size
    warmup_step = cfg.num_image // total_batch_size  # 1 epoch warmup
    total_step = cfg.num_image // total_batch_size * num_epochs

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt, warmup_iters=warmup_step, total_iters=total_step
    )
    warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*optimizer.step.*')

    amp = torch.amp.GradScaler('cuda', growth_interval=100)
    loss_am = AverageMeter()
    global_step = 0

    logging.info(f"Starting FC recovery for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            with torch.no_grad():
                local_embeddings = backbone(img)

            loss = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(module_partial_fc.parameters(), 5)
                amp.step(opt)
                amp.update()
                opt.zero_grad()
                lr_scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module_partial_fc.parameters(), 5)
                opt.step()
                opt.zero_grad()
                lr_scheduler.step()

            loss_am.update(loss.item(), 1)
            if global_step % 100 == 0 and rank == 0:
                logging.info(
                    f"[Epoch {epoch}/{num_epochs}] Step {global_step}: "
                    f"Loss={loss_am.avg:.4f}, LR={lr_scheduler.get_last_lr()[0]:.6f}"
                )
                loss_am.reset()

        logging.info(f"Epoch {epoch} complete.")

    # Save sub-center FC weights (all ranks)
    fc_path = os.path.join(cfg.output, f"subcenter_fc_gpu_{rank}.pt")
    torch.save({
        "weight": module_partial_fc.weight.data.cpu(),
        "num_local": module_partial_fc.num_local,
        "class_start": module_partial_fc.class_start,
        "num_subcenters": num_subcenters,
        "embedding_size": cfg.embedding_size,
        "num_classes": cfg.num_classes,
    }, fc_path)
    logging.info(f"Saved sub-center FC weights: {fc_path}")

    distributed.barrier()
    if rank == 0:
        logging.info("FC recovery complete! Sub-center weights saved.")
        logging.info("You can now run drop_subcenter.py to clean the dataset.")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Recover sub-center FC weights from a trained backbone")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Path to trained backbone .pt file")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train FC head (default: 3)")
    main(parser.parse_args())
