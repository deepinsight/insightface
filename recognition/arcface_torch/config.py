from easydict import EasyDict as edict

config = edict()
config.dataset = "ms1m-retinaface-t2"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.1  # batch size is 512
config.output = "ms1mv3_arcface_r50"

if config.dataset == "emore":
    config.rec = "/train_tmp/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 16
    config.warmup_epoch = -1
    config.val_targets = ["lfw", ]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "ms1m-retinaface-t2":
    config.rec = "/train_tmp/ms1m-retinaface-t2"
    config.num_classes = 91180
    config.num_epoch = 25
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [11, 17, 22] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "glint360k":
    config.rec = "/train_tmp/glint360k"
    config.num_classes = 360232
    config.num_image = 17091657
    config.num_epoch = 20
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [8, 12, 15, 18] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "/train_tmp/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = "forget"
    config.num_epoch = 34
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func

