#!/usr/bin/env python
"""One-command launcher for arcface_torch (一键启动脚本).

    python launch.py                 # interactive: asks for dataset, backbone, loss, weights, output ...
    python launch.py --rec /data/webface42m --network r100 --loss arcface --output work_dirs/r100 --yes

It inspects the RecordIO dataset, recommends batch size / PartialFC sample rate / learning rate for the
detected GPUs, writes `configs/launch_<name>.py`, saves the exact torchrun command to
`<output>/launch_cmd.sh`, and starts training with torchrun.
"""
import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- model zoo
BACKBONES = [
    ("r18", "cnn"), ("r34", "cnn"), ("r50", "cnn"), ("r100", "cnn"), ("r200", "cnn"), ("r2060", "cnn"),
    ("mbf", "mbf"), ("mbf_large", "mbf"),
    ("vit_t", "vit"), ("vit_s", "vit"), ("vit_b_dp005_mask_005", "vit"), ("vit_l_dp005_mask_005", "vit"), ("vit_h", "vit"),
    ("mambavision_t", "mamba"), ("mambavision_s", "mamba"), ("mambavision_b", "mamba"), ("mambavision_l", "mamba"),
    ("transface_vit_b", "transface"), ("transface_vit_l", "transface"),
    ("transface_pp_vit_s", "transface_pp"), ("transface_pp_vit_b", "transface_pp"),
]
FAMILY = dict(BACKBONES)
LOSSES = ["arcface", "cosface", "adaface"]

# per-GPU batch size that fits comfortably on a 141 GB H200 at fp16 (scaled for smaller GPUs)
_BATCH_141GB = {
    "r18": 512, "r34": 512, "r50": 512, "r100": 512, "r200": 256, "r2060": 64, "mbf": 1024, "mbf_large": 1024,
    "vit_t": 512, "vit_s": 512, "vit_b_dp005_mask_005": 256, "vit_l_dp005_mask_005": 192, "vit_h": 128,
    "mambavision_t": 512, "mambavision_s": 512, "mambavision_b": 256, "mambavision_l": 192,
    "transface_vit_b": 256, "transface_vit_l": 128, "transface_pp_vit_s": 256, "transface_pp_vit_b": 128,
}
# (optimizer, base lr, reference total batch for that lr, weight decay, epochs, warmup) - from the official configs
_RECIPE = {
    "cnn":          ("sgd",   0.1,  1024, 5e-4, 20, 2),
    "mbf":          ("sgd",   0.1,  1024, 1e-4, 20, 2),
    "vit":          ("adamw", 1e-3, 2048, 0.1,  40, 4),
    "mamba":        ("adamw", 1e-3, 2048, 0.05, 30, 3),
    "transface":    ("adamw", 1e-3, 2048, 0.1,  35, 3),
    "transface_pp": ("adamw", 1e-3, 2048, 0.1,  20, 2),
}
_MARGIN = {"arcface": (1.0, 0.5, 0.0), "cosface": (1.0, 0.0, 0.4), "adaface": (1.0, 0.5, 0.0)}


# ----------------------------------------------------------------------------- helpers
def inspect_dataset(rec_dir):
    """num_image / num_classes / size of an insightface RecordIO directory (reads two records)."""
    import mxnet as mx
    import numpy as np
    rec_path, idx_path = os.path.join(rec_dir, "train.rec"), os.path.join(rec_dir, "train.idx")
    if not (os.path.exists(rec_path) and os.path.exists(idx_path)):
        raise FileNotFoundError(f"{rec_dir} must contain train.rec and train.idx")
    rec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
    header, _ = mx.recordio.unpack(rec.read_idx(0))
    keys = list(rec.keys)
    if header.flag > 0:
        num_image = int(header.label[0]) - 1
    else:
        num_image = len(keys)
    # insightface records are sorted by label -> the last image carries the largest label
    last_key = max(k for k in keys if k != 0)
    h, _ = mx.recordio.unpack(rec.read_idx(last_key))
    label = h.label if np.isscalar(h.label) else h.label[0]
    num_classes = int(label) + 1
    prop = os.path.join(rec_dir, "property")
    if os.path.exists(prop):                                   # "num_classes,112,112"
        try:
            num_classes = int(open(prop).read().split(",")[0])
        except ValueError:
            pass
    rec.close()
    return {"num_image": num_image, "num_classes": num_classes, "rec_bytes": os.path.getsize(rec_path)}


def detect_gpus():
    try:
        import torch
        n = torch.cuda.device_count()
        mem = [torch.cuda.get_device_properties(i).total_memory / 2 ** 30 for i in range(n)]
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        return n, (min(mem) if mem else 0.0), names
    except Exception:  # noqa: BLE001
        return 0, 0.0, []


def _round32(x):
    return max(32, int(x) // 32 * 32)


def recommend(network, loss, num_gpus, gpu_mem_gb, num_classes, num_image):
    """Recommended training settings for this backbone / loss on the given hardware."""
    assert network in FAMILY, f"unknown backbone {network}"
    assert loss in LOSSES, loss
    fam = FAMILY[network]
    optimizer, base_lr, ref_total, wd, epochs, warmup = _RECIPE[fam]
    batch = _round32(_BATCH_141GB[network] * min(1.0, gpu_mem_gb / 141.0))
    total = batch * num_gpus
    lr = base_lr * total / ref_total
    if optimizer == "sgd":
        lr = min(lr, 0.4)
    if num_classes >= 1_000_000:
        sample_rate = 0.3           # WebFace42M-scale: PartialFC r=0.3 (official configs), no accuracy loss, 3x faster FC
    elif num_classes >= 200_000:
        sample_rate = 0.5
    else:
        sample_rate = 1.0
    if fam == "transface_pp":
        script = "train_transface_pp.py"
    elif fam == "transface":
        script = "train_transface.py"
    elif loss == "adaface":
        script = "train_adaface.py"
    else:
        script = "train_v2.py"
    try:
        import nvidia.dali  # noqa: F401
        dali = True
    except ImportError:
        dali = False
    r = dict(network=network, loss=loss, script=script, optimizer=optimizer, lr=lr, weight_decay=wd,
             batch_size=batch, total_batch=total, sample_rate=sample_rate, num_epoch=epochs,
             warmup_epoch=warmup, fp16=True, dali=dali, num_gpus=num_gpus, num_classes=num_classes,
             num_image=num_image, margin_list=_MARGIN[loss], embedding_size=512)
    if fam == "transface":
        r.update(dpap_prob=0.2, dpap_topk=7, dpap_alpha=1.0, ehsm=True, ehsm_gamma=1.0)
    if fam == "transface_pp":
        r.update(byte_format="tiff", use_topology=True, tibc_prob=0.3, ehsm=True, ehsm_gamma=1.0)
    return r


def render_config(r, rec, output, pretrained=None, resume=False, resume_from=None, resume_epoch=None,
                  num_workers=8, dali=None, seed=2048, val_targets=()):
    dali = r["dali"] if dali is None else dali
    lines = [
        "# generated by launch.py on " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "from easydict import EasyDict as edict",
        "",
        "config = edict()",
        f'config.network = "{r["network"]}"',
        f'config.loss = "{r["loss"]}"',
        f"config.margin_list = {tuple(r['margin_list'])}",
        f'config.output = "{output}"',
        f"config.pretrained = {pretrained!r}",
        f"config.resume = {bool(resume)}",
        f"config.resume_from = {resume_from!r}",
        f"config.resume_epoch = {resume_epoch!r}",
        "config.save_all_states = True",
        "config.keep_epoch_checkpoints = True",
        "",
        f"config.embedding_size = {r['embedding_size']}",
        f"config.sample_rate = {r['sample_rate']}",
        "config.interclass_filtering_threshold = 0",
        f"config.fp16 = {r['fp16']}",
        f"config.batch_size = {r['batch_size']}   # per GPU; total = {r['total_batch']} on {r['num_gpus']} GPUs",
        f'config.optimizer = "{r["optimizer"]}"',
        f"config.lr = {r['lr']}",
        "config.momentum = 0.9",
        f"config.weight_decay = {r['weight_decay']}",
        f"config.num_epoch = {r['num_epoch']}",
        f"config.warmup_epoch = {r['warmup_epoch']}",
        f"config.dali = {dali}",
        "config.dali_aug = False",
        f"config.num_workers = {num_workers}",
        f"config.seed = {seed}",
        "config.verbose = 10000",
        "config.frequent = 10",
        "config.gradient_acc = 1",
        "",
        f'config.rec = "{rec}"',
        f"config.num_classes = {r['num_classes']}",
        f"config.num_image = {r['num_image']}",
        f"config.val_targets = {list(val_targets)}",
        "",
        "# AdaFace",
        "config.adaface_m = 0.4",
        "config.adaface_h = 0.333",
        "config.adaface_s = 64.0",
        "config.adaface_t_alpha = 0.01",
    ]
    extras = {k: r[k] for k in ("dpap_prob", "dpap_topk", "dpap_alpha", "ehsm", "ehsm_gamma",
                                "byte_format", "use_topology", "tibc_prob") if k in r}
    if extras:
        lines.append("")
        lines.append("# TransFace / TransFace++")
        lines += [f"config.{k} = {v!r}" for k, v in extras.items()]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- interactive UI
def _ask(prompt, default=None, choices=None, cast=str, yes=False):
    while True:
        shown = f" [{default}]" if default is not None else ""
        if yes and default is not None:
            return default
        ans = input(f"{prompt}{shown}: ").strip()
        if not ans:
            if default is not None:
                return default
            print("  请输入 / required"); continue
        if choices and ans not in choices:
            print(f"  可选: {', '.join(choices)}"); continue
        try:
            return cast(ans)
        except ValueError:
            print("  无效输入 / invalid"); continue


def _bool(s):
    if isinstance(s, bool):
        return s
    return str(s).lower() in ("1", "y", "yes", "true", "t")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", help="dataset dir containing train.rec / train.idx")
    ap.add_argument("--network", choices=[b for b, _ in BACKBONES])
    ap.add_argument("--loss", choices=LOSSES)
    ap.add_argument("--output", help="work dir for logs / checkpoints")
    ap.add_argument("--pretrained", help="model.pt to initialise the backbone from")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --output")
    ap.add_argument("--resume-from", help="resume from another run dir / '{rank}' pattern")
    ap.add_argument("--resume-epoch", type=int)
    ap.add_argument("--gpus", help="CUDA device ids, e.g. 0,1,2,3 (default: all)")
    ap.add_argument("--batch-size", type=int); ap.add_argument("--sample-rate", type=float)
    ap.add_argument("--lr", type=float); ap.add_argument("--num-epoch", type=int)
    ap.add_argument("--num-classes", type=int); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dali", choices=["on", "off"]); ap.add_argument("--copy-to-shm", action="store_true",
                    help="copy train.rec/idx into /dev/shm first (needs enough RAM)")
    ap.add_argument("--name", help="config name (configs/launch_<name>.py)")
    ap.add_argument("--yes", "-y", action="store_true", help="accept all defaults, no prompts")
    ap.add_argument("--dry-run", action="store_true", help="write the config and command, do not launch")
    a = ap.parse_args()
    yes = a.yes

    print("=" * 72)
    print(" arcface_torch 一键训练启动器  |  one-command training launcher")
    print("=" * 72)
    n_gpu, gpu_mem, names = detect_gpus()
    gpu_ids = a.gpus or ",".join(str(i) for i in range(n_gpu))
    n_use = len(gpu_ids.split(",")) if gpu_ids else 0
    print(f"检测到 GPU: {n_gpu} x {names[0] if names else '?'} ({gpu_mem:.0f} GB)  -> 使用 {gpu_ids or '无'}")
    if n_use == 0:
        sys.exit("no GPU found / 未检测到 GPU")

    # 1. dataset
    rec = a.rec or _ask("1) 训练数据集目录 (含 train.rec/train.idx) / dataset dir", yes=yes)
    info = inspect_dataset(rec)
    num_classes = a.num_classes or info["num_classes"]
    print(f"   数据集: {info['num_image']:,} 张图, {num_classes:,} 个 ID, {info['rec_bytes'] / 2 ** 30:.1f} GB")
    if not a.num_classes and not yes:
        num_classes = _ask("   类别数 num_classes (回车确认自动检测值)", default=num_classes, cast=int)

    # 2. backbone
    if a.network:
        network = a.network
    else:
        print("2) 选择 backbone:")
        for i, (b, fam) in enumerate(BACKBONES, 1):
            print(f"   {i:2d}. {b:22s} ({fam})")
        idx = _ask("   编号", default=4, cast=int, yes=yes)
        network = BACKBONES[idx - 1][0]
    # 3. loss
    loss = a.loss or _ask("3) 损失函数 / loss", default="arcface", choices=LOSSES, yes=yes)
    # 4. weights
    pretrained, resume, resume_from, resume_epoch = a.pretrained, a.resume, a.resume_from, a.resume_epoch
    if not (a.pretrained or a.resume or a.resume_from) and not yes:
        mode = _ask("4) 初始化: 1=从头训练  2=加载预训练 backbone 权重(model.pt)  3=从 checkpoint 断点续训",
                    default="1", choices=["1", "2", "3"])
        if mode == "2":
            pretrained = _ask("   model.pt 路径")
        elif mode == "3":
            resume = True
            resume_from = _ask("   checkpoint 所在目录 (回车=输出目录)", default="") or None
            ep = _ask("   指定 epoch (回车=最新)", default="")
            resume_epoch = int(ep) if ep else None
    # 5. output
    default_out = f"work_dirs/{network}_{loss}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    output = a.output or _ask("5) 输出目录 / output dir", default=default_out, yes=yes)

    # 6. recommended settings
    r = recommend(network, loss, n_use, gpu_mem, num_classes, info["num_image"])
    if a.batch_size: r["batch_size"] = a.batch_size; r["total_batch"] = a.batch_size * n_use
    if a.sample_rate is not None: r["sample_rate"] = a.sample_rate
    if a.lr is not None: r["lr"] = a.lr
    if a.num_epoch: r["num_epoch"] = a.num_epoch
    dali = r["dali"] if a.dali is None else (a.dali == "on")
    print("6) 推荐设置 / recommended settings:")
    print(f"   脚本 {r['script']} | 优化器 {r['optimizer']} lr {r['lr']:.4g} wd {r['weight_decay']}")
    print(f"   batch {r['batch_size']}/GPU x {n_use} = {r['total_batch']} | PartialFC sample_rate {r['sample_rate']}")
    print(f"   epochs {r['num_epoch']} (warmup {r['warmup_epoch']}) | fp16 {r['fp16']} | DALI {dali} | workers {a.num_workers}")
    if not yes and not _bool(_ask("   接受以上设置? / accept", default="y")):
        r["batch_size"] = _ask("   batch_size / GPU", default=r["batch_size"], cast=int); r["total_batch"] = r["batch_size"] * n_use
        r["sample_rate"] = _ask("   sample_rate", default=r["sample_rate"], cast=float)
        r["lr"] = _ask("   lr", default=r["lr"], cast=float)
        r["num_epoch"] = _ask("   num_epoch", default=r["num_epoch"], cast=int)
        dali = _bool(_ask("   DALI (y/n)", default="y" if dali else "n"))

    # 7. dataset in RAM
    shm_free = shutil.disk_usage("/dev/shm").free if os.path.isdir("/dev/shm") else 0
    idx_bytes = os.path.getsize(os.path.join(rec, "train.idx"))
    if a.copy_to_shm or (not yes and shm_free > 1.2 * (info["rec_bytes"] + idx_bytes) and not rec.startswith("/dev/shm")
                         and _bool(_ask(f"7) 把数据集复制到内存盘 /dev/shm (可用 {shm_free / 2 ** 30:.0f} GB, 需要 "
                                        f"{(info['rec_bytes'] + idx_bytes) / 2 ** 30:.0f} GB)? 消除磁盘随机读瓶颈", default="n"))):
        dst = os.path.join("/dev/shm", os.path.basename(os.path.normpath(rec)))
        os.makedirs(dst, exist_ok=True)
        for f in ("train.idx", "train.rec"):
            if not os.path.exists(os.path.join(dst, f)):
                print(f"   copying {f} -> {dst} ..."); shutil.copyfile(os.path.join(rec, f), os.path.join(dst, f))
        rec = dst

    # write config + command
    name = a.name or re.sub(r"[^A-Za-z0-9_]", "_", os.path.basename(os.path.normpath(output)))
    cfg_path = os.path.join(ROOT, "configs", f"launch_{name}.py")
    with open(cfg_path, "w") as f:
        f.write(render_config(r, rec=rec, output=output, pretrained=pretrained, resume=resume,
                              resume_from=resume_from, resume_epoch=resume_epoch, num_workers=a.num_workers, dali=dali))
    port = 12000 + (abs(hash(name)) % 20000)
    cmd = (f"cd {ROOT} && CUDA_VISIBLE_DEVICES={gpu_ids} {sys.executable} -m torch.distributed.run "
           f"--nproc_per_node={n_use} --master_port={port} {r['script']} configs/launch_{name}.py")
    os.makedirs(output if os.path.isabs(output) else os.path.join(ROOT, output), exist_ok=True)
    out_abs = output if os.path.isabs(output) else os.path.join(ROOT, output)
    with open(os.path.join(out_abs, "launch_cmd.sh"), "w") as f:
        f.write("#!/bin/bash\n" + cmd + "\n")
    print("-" * 72)
    print(f"配置已写入 {cfg_path}\n启动命令 (已保存到 {out_abs}/launch_cmd.sh):\n  {cmd}")
    print("-" * 72)
    if a.dry_run:
        return
    if yes or _bool(_ask("现在启动? / launch now", default="y")):
        os.execvp("bash", ["bash", "-c", cmd])


if __name__ == "__main__":
    main()
