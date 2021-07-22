from easydict import EasyDict as edict

# configs for test speed

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512

config.rec = "synthetic"
config.num_classes = 100 * 10000
config.num_epoch = 30
config.warmup_epoch = -1
config.decay_epoch = [10, 16, 22]
config.val_targets = []
