from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "cosface"
config.network = "r18"
config.resume = False
config.output = None
config.embedding_size = 512
config.partial_fc = 1
config.sample_rate = 0.1
config.model_parallel = True
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512

config.dataset = "glint360k"
config.ofrecord_path = "/train_tmp/glint360k/"
config.ofrecord_part_num = 200
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 20
config.warmup_epoch = -1
config.decay_epoch = [8, 12, 15, 18]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
