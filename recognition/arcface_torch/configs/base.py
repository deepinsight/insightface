from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = False
config.output = "ms1mv3_arcface_r50"

config.embedding_size = 512
config.sample_rate = 1
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512
config.dali = False
config.verbose = 2000
config.frequent = 10
config.score = None

# if config.dataset == "emore":
#     config.rec = "/train_tmp/faces_emore"
#     config.num_classes = 85742
#     config.num_image = 5822653
#     config.num_epoch = 16
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", ]

# elif config.dataset == "ms1m-retinaface-t1":
#     config.rec = "/train_tmp/ms1m-retinaface-t1"
#     config.num_classes = 93431
#     config.num_image = 5179510
#     config.num_epoch = 25
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

# elif config.dataset == "glint360k":
#     config.rec = "/train_tmp/glint360k"
#     config.num_classes = 360232
#     config.num_image = 17091657
#     config.num_epoch = 20
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
