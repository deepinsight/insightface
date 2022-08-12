from easydict import EasyDict as edict

config = edict()

config.dataset = "wcpa"
config.root_dir = '/data/insightface/wcpa'
config.cache_dir = './cache_align'
#config.num_classes = 617970
#config.num_classes = 2000000
#config.num_classes = 80000000
#config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
#config.val_targets = ["lfw"]
#config.val_targets = []
config.verbose = 20000

#config.network = 'resnet34d'
config.network = 'resnet_jmlr'
config.input_size = 256
#config.width_mult = 1.0
#config.dropout = 0.0
#config.loss = 'cosface'
#config.embedding_size = 512
#config.sample_rate = 0.2
config.fp16 = 0
config.tf32 = True
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.1  # lr when batch size is 512

config.aug_modes = ['1']

config.num_epochs = 40
config.warmup_epochs = 1
config.max_warmup_steps = 1000

#def lr_step_func(epoch):
#    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
#        [m for m in [20, 30, 38] if m - 1 <= epoch])
#config.lr_func = lr_step_func

config.task = 0
config.save_every_epochs = False


config.lossw_verts3d = 16.0

config.align_face = True

config.use_trainval = True
#config.use_rtloss = True

config.loss_bone3d = True
config.lossw_bone3d = 2.0

