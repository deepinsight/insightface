from easydict import EasyDict as edict

config = edict()
config.dataset = "ms1mv3"
config.fp16 = True
config.batch_size = 128
config.vpl = {'start_iters': 8000, 'allowed_delta': 200, 'lambda': 0.15, 'mode': 0, 'momentum': False}

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 25

def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [11, 17, 22] if m - 1 <= epoch])
config.lr_func = lr_step_func

