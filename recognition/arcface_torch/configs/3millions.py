from easydict import EasyDict as edict

# configs for test speed

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "mbf"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 512 # total_batch_size = batch_size * num_gpus
config.lr = 0.1  # batch size is 512

config.rec = "synthetic"
config.num_classes = 30 * 10000
config.num_image = 100000
config.num_epoch = 30
config.warmup_epoch = -1
config.val_targets = []
