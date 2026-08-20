"""Smoke-test config driven by environment variables.

Used by scripts/smoke_train_all.py to exercise every backbone with both the
ArcFace (CombinedMarginLoss) and AdaFace losses on a small RecordIO subset.

    SMOKE_NETWORK   backbone name understood by backbones.get_model (default r50)
    SMOKE_LOSS      "arcface" | "adaface"                             (default arcface)
    SMOKE_REC       RecordIO directory                                (default webface260m_mock_100mb)
    SMOKE_OUTPUT    output/work dir                                   (default work_dirs/smoke/<net>_<loss>)
    SMOKE_EPOCHS    number of epochs                                  (default 3)
    SMOKE_BS        per-GPU batch size                                (default 128)
    SMOKE_WORKERS   dataloader workers per rank                       (default 4)
    SMOKE_DALI      "1" to use the NVIDIA DALI GPU pipeline            (default 0)
    SMOKE_DALI_AUG  "1" to enable DALI-side augmentation               (default 0)
    SMOKE_LR        override the learning rate                        (default per family)
    SMOKE_SAVE_ALL  "1" to save full resumable checkpoints each epoch   (default 0)
    SMOKE_RESUME    "1" to resume from SMOKE_OUTPUT/checkpoint_gpu_*.pt (default 0)
    SMOKE_BYTE_FORMAT  TransFace++ input bytes: tiff|png|fhwc|fchw       (default tiff)
"""
import os

from easydict import EasyDict as edict

config = edict()

network = os.environ.get("SMOKE_NETWORK", "r50")
loss = os.environ.get("SMOKE_LOSS", "arcface")

config.network = network
config.loss = loss  # consumed by train_transface.py; informational elsewhere
config.resume = os.environ.get("SMOKE_RESUME", "0") == "1"
config.save_all_states = os.environ.get("SMOKE_SAVE_ALL", "0") == "1"
config.keep_epoch_checkpoints = config.save_all_states
config.resume_epoch = int(os.environ["SMOKE_RESUME_EPOCH"]) if os.environ.get("SMOKE_RESUME_EPOCH") else None
config.resume_from = os.environ.get("SMOKE_RESUME_FROM") or None
config.output = os.environ.get("SMOKE_OUTPUT", os.path.join("work_dirs", "smoke", f"{network}_{loss}"))

config.embedding_size = 512
config.sample_rate = 1.0
config.interclass_filtering_threshold = 0
config.margin_list = (1.0, 0.5, 0.0)
config.fp16 = True
config.batch_size = int(os.environ.get("SMOKE_BS", 128))
config.num_workers = int(os.environ.get("SMOKE_WORKERS", 4))
config.dali = os.environ.get("SMOKE_DALI", "0") == "1"
config.dali_aug = os.environ.get("SMOKE_DALI_AUG", "0") == "1"
config.gradient_acc = 1
config.seed = 2048

# Optimiser family follows the reference configs: CNNs use SGD, ViT/Mamba use AdamW.
if network.startswith(("r", "mbf")):
    config.optimizer = "sgd"
    config.lr = 0.1
    config.momentum = 0.9
    config.weight_decay = 5e-4
else:
    config.optimizer = "adamw"
    config.lr = 1e-4
    config.weight_decay = 0.05 if network.startswith("mamba") else 0.1

if os.environ.get("SMOKE_LR"):
    config.lr = float(os.environ["SMOKE_LR"])

# AdaFace hyperparameters (only used when loss == "adaface")
config.adaface_m = 0.4
config.adaface_h = 0.333
config.adaface_s = 64.0
config.adaface_t_alpha = 0.01

# TransFace DPAP (only used by train_transface.py)
config.dpap_prob = 0.2
config.dpap_topk = 7
config.dpap_alpha = 1.0

# TransFace++ (only used by train_transface_pp.py)
config.byte_format = os.environ.get("SMOKE_BYTE_FORMAT", "tiff")
config.use_topology = True
config.tibc_prob = 0.3
config.ehsm = True
config.ehsm_gamma = 1.0

# Dataset: ~100MB cut of the WebFace260M mock set (see scripts/smoke_train_all.py)
config.rec = os.environ.get("SMOKE_REC", "/data/fu.dai/face_dataset/webface260m_mock_100mb")
config.num_classes = int(os.environ.get("SMOKE_NUM_CLASSES", 544))
config.num_image = int(os.environ.get("SMOKE_NUM_IMAGE", 37173))
config.num_epoch = int(os.environ.get("SMOKE_EPOCHS", 3))
config.warmup_epoch = 0
config.val_targets = []  # no verification bins in the mock set

config.frequent = 10      # log loss every 10 steps
config.verbose = 10 ** 9  # never run verification
