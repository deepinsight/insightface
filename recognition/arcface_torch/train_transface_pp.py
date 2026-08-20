"""TransFace++ training (TPAMI 2025): face recognition directly from image bytes.

Differences from train_v2.py:
  * the backbone (`transface_pp_vit_*`) consumes the image *bytes* produced by
    `image_to_bytes(img, cfg.byte_format)` (fhwc / fchw raw pixels, png / tiff
    file bytes) instead of the decoded pixels;
  * EHSM: the per-token feature entropy returned by the backbone re-weights each
    sample's loss (cfg.ehsm, cfg.ehsm_gamma);
  * cfg.loss selects ArcFace (CombinedMarginLoss) or AdaFace.
"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from backbones.transface_pp import image_to_bytes
from dataset import get_dataloader
from losses import AdaFaceLoss, CombinedMarginLoss, ehsm_sample_weight
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2, PartialFC_V2_AdaFace
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_checkpoint import load_checkpoint, load_pretrained_backbone, save_checkpoint
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

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

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                sync_tensorboard=True,
                resume=cfg.wandb_resume,
                name=run_name,
                notes=cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
    )

    loss_type = cfg.get("loss", "arcface")
    assert loss_type in ("arcface", "adaface"), loss_type
    use_adaface = loss_type == "adaface"
    byte_format = cfg.get("byte_format", "tiff")
    use_ehsm = cfg.get("ehsm", True)

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16,
        num_features=cfg.embedding_size,
        norm_output=use_adaface,
        byte_format=byte_format,
        use_topology=cfg.get("use_topology", True),
        tibc_prob=cfg.get("tibc_prob", 0.3)).cuda()
    num_bytes = backbone.num_bytes

    if cfg.get("pretrained"):
        load_pretrained_backbone(backbone, cfg.pretrained)

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank],
        bucket_cap_mb=16, find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    backbone._set_static_graph()

    if use_adaface:
        margin_loss = AdaFaceLoss(
            m=cfg.adaface_m,
            h=cfg.adaface_h,
            s=cfg.adaface_s,
            t_alpha=cfg.adaface_t_alpha,
        )
        pfc_cls = PartialFC_V2_AdaFace
    else:
        margin_loss = CombinedMarginLoss(
            64,
            cfg.margin_list[0],
            cfg.margin_list[1],
            cfg.margin_list[2],
            cfg.interclass_filtering_threshold,
        )
        pfc_cls = PartialFC_V2

    module_partial_fc = pfc_cls(
        margin_loss, cfg.embedding_size, cfg.num_classes,
        cfg.sample_rate, False)
    module_partial_fc.train().cuda()
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    start_epoch, global_step = load_checkpoint(
        cfg, rank, backbone, module_partial_fc, opt, lr_scheduler, amp)

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))
    logging.info(": byte_format              %s (%d bytes per image)", byte_format, num_bytes)

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger=wandb_logger,
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
    )

    loss_am = AverageMeter()

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            # --- TransFace++: the model sees bytes, not pixels ---
            byte_seq = image_to_bytes(img, byte_format, num_bytes)
            outputs = backbone(byte_seq)          # (emb, [norm,] patch_entropy)
            local_embeddings, patch_entropy = outputs[0], outputs[-1]
            local_norms = outputs[1] if use_adaface else None
            sample_weight = ehsm_sample_weight(patch_entropy, cfg.get("ehsm_gamma", 1.0)) if use_ehsm else None
            # -----------------------------------------------------

            if use_adaface:
                loss: torch.Tensor = module_partial_fc(local_embeddings, local_norms, local_labels, sample_weight)
            else:
                loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, sample_weight)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch,
                    })

                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16,
                                 lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            save_checkpoint(cfg, rank, epoch, global_step, backbone, module_partial_fc,
                            opt, lr_scheduler, amp)

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed TransFace++ Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
