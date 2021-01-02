from easydict import EasyDict as edict

config = edict()
config.dataset = "glint360k"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.1
config.output = "tmp_models"

if config.dataset == "emore":
    config.rec = "/train_tmp/faces_emore"
    config.num_classes = 85742
    config.num_epoch = 16

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "glint360k":
    config.rec = "/train_tmp/glint360k"
    config.num_classes = 360232
    config.num_image = 17091657
    config.num_epoch = 17
    config.warmup_epoch = -1

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [6, 10, 14] if m - 1 <= epoch])
    config.lr_func = lr_step_func
