from easydict import EasyDict as edict
import math
import numpy as np
# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "cosface"
config.network = "r50"
config.resume = False
config.output = "ms1mv3_arcface_r50"

config.dataset = "ms1m-retinaface-t1"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512
config.model_load_dir = ''

config.val_batch_size = 10

config.node_ips = ["192.168.1.13"]
config.num_nodes = 1
config.device_num_per_node = 1
config.model_parallel = 1
config.partial_fc = 0

config.use_synthetic_data = False


config.fc_type = "FC"
config.nccl_fusion_threshold_mb = 16
config.nccl_fusion_max_ops = 64
config.val_dataset_dir = "/train_tmp/glint360k/val"


config.part_name_prefix = "part-"
config.part_name_suffix_length = 5
config.train_data_part_num = 16
config.shuffle = True


if config.dataset == "emore":
    config.ofrecord_path = "/train_tmp/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 16
    config.warmup_epoch = -1
    config.decay_epoch = [8, 14, ]
    config.val_targets = ["lfw", ]
    config.train_data_part_num = 32

elif config.dataset == "ms1m-retinaface-t1":
    config.ofrecord_path = "/dev/shm/ms1m-retinaface-t1/ofrecord"
    config.num_classes = 93432
    config.num_image = 5179510
    config.num_epoch = 25
    config.warmup_epoch = -1
    config.decay_epoch = [11, 17, 22]
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
    config.train_data_part_num = 32

elif config.dataset == "glint360k":
    config.ofrecord_path = "/train_tmp/glint360k"
    config.num_classes = 360232
    config.num_image = 17091657
    config.num_epoch = 20
    config.warmup_epoch = -1
    config.decay_epoch = [8, 12, 15, 18]
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

elif config.dataset == "webface":
    config.ofrecord_path = "/train_tmp/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = "forget"
    config.num_epoch = 34
    config.warmup_epoch = -1
    config.decay_epoch = [20, 28, 32]
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
