from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "cosface"
config.network = "r50"
config.resume = False
config.output = "lazy_r50"
config.embedding_size = 512
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512
config.model_parallel = True
config.partial_fc = 1
config.sample_rate = 1.0
config.device_num_per_node = 8


config.ofrecord_path = "/dev/shm/ms1m-retinaface-t1/ofrecord/train"
config.eval_ofrecord_path = "/dev/shm/ms1m-retinaface-t1/ofrecord/val"
config.num_classes = 93432
config.num_image = 5179510
config.train_data_part_num = 32

config.num_epoch = 25
config.warmup_epoch = -1
config.decay_epoch = [10, 16, 22]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]


config.node_ips = ["192.168.1.13"]
config.num_nodes = 1
