from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "mbf"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 2e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 25
config.warmup_epoch = -1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
        [m for m in [11, 17, 22] if m - 1 <= epoch])
config.lr_func = lr_step_func
