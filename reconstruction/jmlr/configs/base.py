from easydict import EasyDict as edict
import numpy as np

config = edict()
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = 0
config.tf32 = False
config.backbone_wd = None
config.batch_size = 128
config.clip_grad = None
config.dropout = 0.0
#config.warmup_epoch = -1
config.loss = 'cosface'
config.margin = 0.4
config.hard_margin = False
config.network = 'r50'
config.prelu = True
config.stem_type = ''
config.dropblock = 0.0
config.output = None
config.input_size = 112
config.width_mult = 1.0
config.kaiming_init = True
config.use_se = False
config.aug_modes = []
config.checkpoint_segments = [1, 1, 1, 1]

config.sampling_id = True
config.id_sampling_ratio = None
metric_loss = edict()
metric_loss.enable = False
metric_loss.lambda_n = 0.0
metric_loss.lambda_c = 0.0
metric_loss.lambda_t = 0.0
metric_loss.margin_c = 1.0
metric_loss.margin_t = 1.0
metric_loss.margin_n = 0.4
config.metric_loss = metric_loss

config.opt = 'sgd'
config.lr = 0.1  # when batch size is 512
config.momentum = 0.9
config.weight_decay = 5e-4
config.fc_mom = 0.9

config.warmup_epochs = 0
config.max_warmup_steps = 6000
config.num_epochs = 24


config.resume = False
config.resume_path = None
config.resume_from = None

config.save_every_epochs = True

config.lr_func = None
config.lr_epochs = None
config.save_pfc = False
config.save_onnx = False
config.save_opt = False

config.label_6dof_mean = np.array([-0.018197, -0.017891, 0.025348, -0.005368, 0.001176, -0.532206], dtype=np.float32)  # mean of pitch, yaw, roll, tx, ty, tz
config.label_6dof_std = np.array([0.314015, 0.271809, 0.081881, 0.022173, 0.048839, 0.065444], dtype=np.float32)        # std of pitch, yaw, roll, tx, ty, tz

config.num_verts = 1220
config.flipindex_file = 'cache_align/flip_index.npy'
config.enable_flip = True
config.verts3d_central_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 956, 975, 1022, 1041, 1047, 1048, 1049]

config.task = 0
config.ckpt = None
config.loss_hard = False
config.sampling_hard = False
config.loss_pip = False
config.net_stride = 32
config.loss_bone3d = False
config.loss_bone2d = False

config.lossw_verts3d = 8.0
config.lossw_verts2d = 16.0
config.lossw_bone3d = 10.0
config.lossw_bone2d = 10.0
config.lossw_project = 10.0
config.lossw_eyes3d = 8.0
config.lossw_eyes2d = 16.0

config.align_face = False
config.no_gap = False

config.use_trainval = False

config.project_loss = False

config.use_onenetwork = True

config.use_rtloss = False


config.use_arcface = False


config.eyes = None

