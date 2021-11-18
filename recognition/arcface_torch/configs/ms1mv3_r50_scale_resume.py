from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = True
config.output = "/gpfs/gpfs0/r.kail/insightface/recognition/arcface_torch/work_dirs/ms1mv3_r50"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512

config.rec = "/gpfs/gpfs0/k.fedyanin/space/ms1m"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 25
config.warmup_epoch = -1
config.decay_epoch = [0, 0, 4]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
