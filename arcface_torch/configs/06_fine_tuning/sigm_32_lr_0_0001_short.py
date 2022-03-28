from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface_scale"
config.network = "r50"
config.resume = True
config.source = "/gpfs/gpfs0/k.fedyanin/space/models/scale/source"
config.scale_source = "/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/32/checkpoint.pth"
config.output = "/gpfs/gpfs0/k.fedyanin/space/models/scale/06_fine_tuning/sigm_32_lr_0.0001_short"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.0001  # batch size is 512
config.scale_lr = 0.0001
config.freeze_backbone = False
config.scale_predictor_sizes = [25088, 512, 1]
config.scale_batch_norm = True
config.scale_activation = "sigm_mul"
config.scale_coefficient = 32.

config.rec = "/gpfs/gpfs0/k.fedyanin/space/ms1m"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 3
config.warmup_epoch = 0.5
config.decay_epoch = [2, ]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
