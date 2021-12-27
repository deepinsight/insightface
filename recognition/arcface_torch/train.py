import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import losses
from backbones import get_model, MLPHead, DummyHead
from dataset import MXFaceDataset, SyntheticDataset, DataLoaderX
from partial_fc import PartialFC
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


def main(args):
    cfg = get_config(args.config)
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group('nccl')
    except KeyError:
        world_size = 1
        rank = 0
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)

    # local_rank = args.local_rank
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    if cfg.rec == "synthetic":
        train_set = SyntheticDataset(local_rank=local_rank)
    else:
        train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)
    scale_predictor = MLPHead(
        num_feats=cfg.scale_predictor_sizes, batch_norm=cfg.scale_batch_norm,
        exponent=cfg.scale_exponent, coefficient=cfg.scale_coefficient, fp16=cfg.fp16).to(local_rank)

    if cfg.resume:
        try:
            backbone_pth = os.path.join(cfg.source, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            if rank == 0:
                logging.info("resume fail, backbone init successfully!")

    if not cfg.freeze_backbone:
        backbone = torch.nn.parallel.DistributedDataParallel(
            module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    for p in scale_predictor.parameters():
        dist.broadcast(p, 0)
    scale_predictor = torch.nn.parallel.DistributedDataParallel(
        module=scale_predictor, broadcast_buffers=False, device_ids=[local_rank])

    if cfg.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
    else:
        backbone.train()
    scale_predictor.train()

    margin_softmax = losses.get_loss(cfg.loss)
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=cfg.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output, source=cfg.source)

    learnable_parameters = []
    if not cfg.freeze_backbone:
        learnable_parameters += list(backbone.parameters())
    if cfg.loss == "arcface_scale":
        learnable_parameters += list(scale_predictor.parameters())

    param_groups = []
    if not cfg.freeze_backbone:
        param_groups.append({
            "params": backbone.parameters(),
            "lr": cfg.lr / 512 * cfg.batch_size * world_size,
            "momentum": 0.9,
            "weight_decay": cfg.weight_decay
        })
    if cfg.loss == "arcface_scale":
        param_groups.append({
            "params": scale_predictor.parameters(),
            "lr": cfg.scale_lr / 512 * cfg.batch_size * world_size,
            "momentum": 0.9,
            "weight_decay": cfg.weight_decay
        })

    opt_backbone = torch.optim.SGD(param_groups)

    if cfg.freeze_backbone:
        for p in module_partial_fc.parameters():
            p.requires_grad = False
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    # print(f"Opt backbone parameters {sum(p.numel() for p in opt_backbone.param_groups)}")
    # print(f"Opt pfc {sum(p.numel() for p in opt_pfc.param_groups)}")

    num_image = len(train_set)
    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch

    def lr_step_func(current_step):
        cfg.decay_step = [x * num_image // total_batch_size for x in cfg.decay_epoch]
        if current_step < cfg.warmup_step:
            return current_step / cfg.warmup_step
        else:
            return 0.1 ** len([m for m in cfg.decay_step if m <= current_step])

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=lr_step_func)

    if not cfg.freeze_backbone:
        scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt_pfc, lr_lambda=lr_step_func)

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    val_target = cfg.val_targets
    callback_verification = CallBackVerification(2000, rank, val_target, cfg.rec, device=torch.device("cuda:0"))
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    start_epoch = 0
    global_step = 0
    grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)

        # callback_verification(global_step, backbone)
        # print("Managed callback verification")

        for step, (img, label) in enumerate(train_loader):
            global_step += 1

            output = backbone(img)
            output.update(scale_predictor(**output))

            features = F.normalize(output["feature"])
            scale = output["scale"]

            if cfg.loss == "arcface_scale":
                x_grad, s_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc, scale=scale)

                if cfg.fp16:
                    if not cfg.freeze_backbone:
                        features.backward(grad_amp.scale(x_grad), retain_graph=True)
                    scale.backward(grad_amp.scale(s_grad))

                    grad_amp.unscale_(opt_backbone)
                    clip_grad_norm_(learnable_parameters, max_norm=5, norm_type=2)
                    grad_amp.step(opt_backbone)
                    grad_amp.update()

                    # for idx, p in enumerate(scale_predictor.parameters()):
                    #     print(f"\t[{local_rank}] scale param no {idx} : {p.shape}, {p.grad is None}, requires : {p.requires_grad} sum : {p.sum().item()}")
                    # for idx, p in enumerate(backbone.parameters()):
                    #     print(f"\t[{local_rank}] backbone no {idx} : {p.shape}, {p.grad is None}, requires : {p.requires_grad} sum : {p.sum().item()}")
                    #     if idx > 3: break

                else:
                    features.backward(x_grad)
                    scale.backward(grad_amp.scale(s_grad))
                    clip_grad_norm_(learnable_parameters, max_norm=5, norm_type=2)
                    opt_backbone.step()

                # for p in backbone.parameters():
                #     print(f"Back : {p.shape} : val : {p[0, 0, 0, 0]}")
                #     break
                # for p in scale_predictor.parameters():
                #     print(f"Scale : {p.shape} : val : {p[0, 0]}")
                #     break
                # for p in module_partial_fc.parameters():
                #     print(f"PFC : {p.shape} : val : {p[0, 0]}")
                #     break


            else:
                x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)

                if not cfg.freeze_backbone:
                    if cfg.fp16:
                        features.backward(grad_amp.scale(x_grad))
                        grad_amp.unscale_(opt_backbone)
                        clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                        grad_amp.step(opt_backbone)
                        grad_amp.update()
                    else:
                        features.backward(x_grad)
                        clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                        opt_backbone.step()

            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, scheduler_backbone.get_last_lr()[0], grad_amp)
            callback_verification(global_step, backbone)
            scheduler_backbone.step()
            if not cfg.freeze_backbone:
                scheduler_pfc.step()

        callback_checkpoint(global_step, backbone, module_partial_fc, scale_predictor=scale_predictor)
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
