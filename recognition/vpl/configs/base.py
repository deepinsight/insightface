from easydict import EasyDict as edict

config = edict()
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = False
config.tf32 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # when batch size is 512
config.warmup_epoch = -1
config.loss = 'arcface'
config.network = 'r50'
config.output = None
config.val_targets = ['lfw', "cfp_fp", "agedb_30"]
config.vpl = {'start_iters': 8000, 'allowed_delta': 200, 'lambda': 0.15, 'mode': -1, 'momentum': False} #mode==-1 disables vpl


