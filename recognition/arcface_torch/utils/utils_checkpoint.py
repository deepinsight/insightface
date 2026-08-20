"""Resumable training checkpoints shared by train_v2.py, train_adaface.py,
train_transface.py and train_transface_pp.py.

Files (per rank, because PartialFC shards the class weights across ranks):
    {output}/checkpoint_gpu_{rank}.pt               latest, overwritten every epoch
    {output}/checkpoint_epoch{E}_gpu_{rank}.pt      kept when cfg.keep_epoch_checkpoints is True

Resume options (all read from the config):
    resume = True                resume from the latest checkpoint in cfg.output
    resume_epoch = E             ... from the epoch-E checkpoint instead
    resume_from = <dir>          ... look in <dir> instead of cfg.output
    resume_from = <pattern>      explicit file; may contain {rank} / {epoch} placeholders
The number of GPUs must match the run that wrote the checkpoint.
"""
import logging
import os
import shutil

import torch

CHECKPOINT_LATEST = "checkpoint_gpu_{rank}.pt"
CHECKPOINT_EPOCH = "checkpoint_epoch{epoch}_gpu_{rank}.pt"


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def save_checkpoint(cfg, rank, epoch, global_step, backbone, module_partial_fc, opt, lr_scheduler,
                    grad_scaler=None):
    """Write the checkpoint for the epoch that just finished (`epoch`, 0-based).

    Stores epoch + 1 as the next epoch to run, matching the original scripts."""
    checkpoint = {
        "epoch": epoch + 1,
        "global_step": global_step,
        "num_epoch": cfg.get("num_epoch"),
        "state_dict_backbone": _unwrap(backbone).state_dict(),
        "state_dict_softmax_fc": module_partial_fc.state_dict(),
        "state_optimizer": opt.state_dict(),
        "state_lr_scheduler": lr_scheduler.state_dict(),
    }
    if grad_scaler is not None:
        checkpoint["state_grad_scaler"] = grad_scaler.state_dict()
    os.makedirs(cfg.output, exist_ok=True)
    latest = os.path.join(cfg.output, CHECKPOINT_LATEST.format(rank=rank))
    torch.save(checkpoint, latest + ".tmp")
    os.replace(latest + ".tmp", latest)                      # never leave a half-written "latest"
    if cfg.get("keep_epoch_checkpoints", False):
        shutil.copyfile(latest, os.path.join(cfg.output, CHECKPOINT_EPOCH.format(epoch=epoch, rank=rank)))
    return latest


def resolve_resume_path(cfg, rank):
    """Path of the checkpoint this rank should resume from, or None when cfg.resume is off."""
    if not cfg.get("resume", False):
        return None
    resume_from = cfg.get("resume_from")
    resume_epoch = cfg.get("resume_epoch")
    base_dir = cfg.output
    if resume_from:
        if "{rank}" in resume_from or "{epoch}" in resume_from:
            return resume_from.format(rank=rank, epoch=resume_epoch)
        if os.path.isdir(resume_from):
            base_dir = resume_from
        else:
            return resume_from
    if resume_epoch is not None:
        return os.path.join(base_dir, CHECKPOINT_EPOCH.format(epoch=resume_epoch, rank=rank))
    return os.path.join(base_dir, CHECKPOINT_LATEST.format(rank=rank))


def load_checkpoint(cfg, rank, backbone, module_partial_fc, opt, lr_scheduler, grad_scaler=None):
    """Restore all training state in place. Returns (start_epoch, global_step); (0, 0) when not resuming."""
    path = resolve_resume_path(cfg, rank)
    if path is None:
        return 0, 0
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume checkpoint not found for rank {rank}: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _unwrap(backbone).load_state_dict(checkpoint["state_dict_backbone"])
    module_partial_fc.load_state_dict(checkpoint["state_dict_softmax_fc"])
    opt.load_state_dict(checkpoint["state_optimizer"])
    lr_scheduler.load_state_dict(checkpoint["state_lr_scheduler"])
    if grad_scaler is not None and "state_grad_scaler" in checkpoint:
        grad_scaler.load_state_dict(checkpoint["state_grad_scaler"])
    saved_num_epoch = checkpoint.get("num_epoch")
    if saved_num_epoch is not None and cfg.get("num_epoch") != saved_num_epoch:
        logging.warning(
            "resuming with num_epoch=%s but the checkpoint was written by a run with num_epoch=%s; "
            "the restored LR scheduler keeps the original total_iters",
            cfg.get("num_epoch"), saved_num_epoch)
    logging.info("resumed from %s: next epoch %d, global_step %d",
                 path, checkpoint["epoch"], checkpoint["global_step"])
    return checkpoint["epoch"], checkpoint["global_step"]


def load_pretrained_backbone(backbone, path):
    """Initialise the backbone from a `model.pt` (a backbone state_dict as written by the training
    scripts, with or without DDP's 'module.' prefix) before training starts.

    Unlike `load_checkpoint` this touches only the backbone weights: tensors whose name or shape
    does not match (e.g. a different embedding head) are skipped and reported. Returns a dict with
    the loaded / skipped / missing key lists."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict_backbone" in state:          # a full checkpoint also works
        state = state["state_dict_backbone"]
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    model = _unwrap(backbone)
    own = model.state_dict()
    usable, skipped = {}, []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            usable[k] = v
        else:
            skipped.append(k)
    missing = [k for k in own if k not in usable]
    model.load_state_dict(usable, strict=False)
    logging.info("pretrained backbone %s: loaded %d tensors, skipped %d (name/shape mismatch), %d left at init",
                 path, len(usable), len(skipped), len(missing))
    if skipped:
        logging.info("  skipped: %s", skipped[:10] + (["..."] if len(skipped) > 10 else []))
    return {"loaded": len(usable), "skipped": skipped, "missing": missing}
