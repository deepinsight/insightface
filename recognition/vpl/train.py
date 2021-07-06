import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import losses
from backbones import get_model
from dataset import MXFaceDataset, DataLoaderX
from torch.utils.data import DataLoader, Dataset
from vpl import VPL
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_dist import concat_all_gather, batch_shuffle_ddp, batch_unshuffle_ddp
from utils.utils_config import get_config


def main(args):
    cfg = get_config(args.config)
    if not cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"

    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    if not os.path.exists(cfg.output) and rank==0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)
    if rank==0:
        logging.info(args)
        logging.info(cfg)

    train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)

    dropout = 0.4 if cfg.dataset == "webface" else 0
    backbone = get_model(cfg.network, dropout=dropout, fp16=cfg.fp16).to(local_rank)
    backbone_onnx = get_model(cfg.network, dropout=dropout, fp16=False)

    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank==0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail, backbone init successfully!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    cfg_vpl = cfg.vpl
    vpl_momentum = cfg_vpl['momentum']
    if vpl_momentum:
        backbone_w = get_model(cfg.network, dropout=dropout, fp16=cfg.fp16).to(local_rank)
        backbone_w.train()
        for param_b, param_w in zip(backbone.module.parameters(), backbone_w.parameters()):
            param_w.data.copy_(param_b.data)
            param_w.requires_grad = False

    margin_softmax = losses.get_loss(cfg.loss)
    module_fc = VPL(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=args.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output,
        cfg = cfg_vpl)
    #print('AAA')

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    #print('AAA')
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_pfc, lr_lambda=cfg.lr_func)

    start_epoch = 0
    total_step = int(len(train_set) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank==0: logging.info("Total Step is: %d" % total_step)

    #for epoch in range(start_epoch, cfg.num_epoch):
    #    _lr = cfg.lr_func(epoch)
    #    logging.info('%d:%f'%(epoch, _lr))

    callback_verification = CallBackVerification(10000, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    use_batch_shuffle = True
    alpha = 0.999
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            global_step += 1
            #img = img.to(memory_format=torch.channels_last)
            features = F.normalize(backbone(img))
            feature_w = None
            if vpl_momentum:
                with torch.no_grad():
                    for param_b, param_w in zip(backbone.module.parameters(), backbone_w.parameters()):
                        param_w.data = param_w.data * alpha + param_b.data * (1. - alpha)
                    if use_batch_shuffle:
                        img_w, idx_unshuffle = batch_shuffle_ddp(img, rank, world_size)

                    feature_w = F.normalize(backbone_w(img_w))
                    if use_batch_shuffle:
                        feature_w = batch_unshuffle_ddp(feature_w, idx_unshuffle, rank, world_size)
                    feature_w = feature_w.detach()

            x_grad, loss_v = module_fc.forward_backward(label, features, opt_pfc, feature_w)
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
            module_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_amp)
            callback_verification(global_step, backbone)
        callback_checkpoint(global_step, backbone, module_fc, backbone_onnx)
        scheduler_backbone.step()
        scheduler_pfc.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace-VPL Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args_ = parser.parse_args()
    main(args_)

