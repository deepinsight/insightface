import os
import os.path as osp
import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()
config.GPU = '0'
config.USE_GT = True
config.NUM_WORKERS = 16
config.DEBUG = False
config.PRINT_INFO = True
config.PRINT_INTERVAL = 10
config.BATCH_SIZE = 1024
config.RANDOM_SEED = 2019

config.OUTPUT_DIR = '../output'
config.LOG_DIR = '../log'
config.DATA_DIR = '../data'

config.DATA = edict()
config.DATA.DATASET_NAME = "h36m"
config.DATA.FRAME_INTERVAL = 1
config.DATA.EXP_TMC = False
config.DATA.EXP_TMC_START = 0
config.DATA.EXP_TMC_DETERMINISTIC = False
config.DATA.EXP_TMC_INTERVAL = 5
config.DATA.NUM_FRAMES = 1
config.DATA.NUM_JOINTS = 17
config.DATA.DIFF_SUFFIX = "diff5"
config.DATA.USE_IDEAL_SCALE = False
config.DATA.USE_EXTRA_DATA = False
config.DATA.USE_RANDOM_DIFF = False
config.DATA.MIN_DIFF_DIST = 5
config.DATA.SCALE_MID_MEAN = 0.714
config.DATA.SCALE_MID_STD = 0.051
config.DATA.NUM_NEIGHBOUR_FRAMES = 1
config.DATA.NUM_NEIGHBOUR_TUPLES = 0
config.DATA.NEIGHBOUR_FRAME_INTERVAL = 1
config.DATA.ONLINE_ROT = False
# config.DATA.INDICES_14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16]
# config.DATA.PARENT_17 = [1, 2, 13, 13, 3, 4, 7, 8, 12, 12, 9, 10, 14, 13, 13, 12, 15] 
# -1 will not be considered in the bone_indices below
# config.DATA.PARENT_14 = [1, 2, 12, 12, 3, 4, 7, 8, -1, -1, 9, 10, 13, -1]
# the bones to be consider, exclude pelvis
# config.DATA.BONES_17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
# config.DATA.BONES_14 = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
# config.DATA.PARENTS = config.DATA.PARENT_17 if config.DATA.NUM_JOINTS == 17 else config.DATA.PARENT_14
# need to be updated
config.DATA.TRAIN_PATH = ''
config.DATA.VALID_PATH = ''
config.DATA.USE_SAME_NORM_2D = True
config.DATA.USE_SAME_NORM_3D = False
config.DATA.USE_2D_GT_SUPERVISION = False

config.NETWORK = edict()
# whether to use the model arch in `a simple yet effective baseline`
config.NETWORK.DIS_USE_SPECTRAL_NORM = False
config.NETWORK.USE_BASELINE = False
config.NETWORK.NUM_CHANNELS = 1024 
config.NETWORK.DROPOUT = 0.25 
config.NETWORK.BN_TRACK = True
config.NETWORK.DIS_USE_BN = False
config.NETWORK.DEPTH_ESTIMATOR_RES_BLOCKS = 1
config.NETWORK.LIFTER_RES_BLOCKS = 4
config.NETWORK.DIS_RES_BLOCKS = 2
config.NETWORK.DIS_TEMP_RES_BLOCKS = 2
config.NETWORK.SCALER_INPUT_SIZE = 37 
config.NETWORK.SCALER_RES_BLOCKS = 1
config.NETWORK.ROTATER_RES_BLOCKS = 1
config.NETWORK.ROTATER_PRE_EULER = True
config.NETWORK.USE_SIMPLE_MODEL = False

config.TRAIN = edict()
config.TRAIN.SCHEDULER_STEP_SIZE = 1 
config.TRAIN.SCHEDULER_GAMMA = 0.95
config.TRAIN.SCALE_ON_3D = False
config.TRAIN.USE_GT_SCALE = False
config.TRAIN.GENERIC_BASELINE = False
config.TRAIN.LOSS_TYPE = "ss"
config.TRAIN.FINETUNE_ROTATER = False
config.TRAIN.USE_BONE_NOSCALE = False
config.TRAIN.USE_CYCLE = False
config.TRAIN.USE_SCALER = False
config.TRAIN.LEARN_SYMMETRY = True
config.TRAIN.USE_ROTATER = False
config.TRAIN.SCALER_MEAN_ONLY = False
# scaler relevant 
config.TRAIN.SCALE_MID_PER_SEQ_OPT = False 
config.TRAIN.SCALE_MID_ACC_STEPS = 1
config.TRAIN.USE_TEMP = False
config.TRAIN.USE_NEW_TEMP = False
config.TRAIN.USE_DIS = True
config.TRAIN.USE_LSTM = False
config.TRAIN.MULTI_NEW_TEMP = False
config.TRAIN.CAT_CURRENT = False
config.TRAIN.PRETRAIN_LIFTER = True 
config.TRAIN.LIFTER_PRETRAIN_PATH = "../output/model.pth.tar"
config.TRAIN.CAMERA_SKELETON_DISTANCE = 10.0
config.TRAIN.USE_NEW_ROT = True
config.TRAIN.RESUME_TRAIN = True
config.TRAIN.SUBNET_CRITICS = 1
config.TRAIN.MAINNET_CRITICS = 1
config.TRAIN.ROTATENET_CRITICS = 1
# in the paper weights are [0.001, 10.0, 1.0]
config.TRAIN.SCALE_LOSS_WEIGHTS = [1.0, 1.0]
# first for the temporal, second for the euler penalty 
config.TRAIN.ROTATE_LOSS_WEIGHTS = [1.0, 0.001]
config.TRAIN.LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 10.0]
# config.TRAIN.LOSS_WEIGHTS = [0.05, 5.0, 1.5]
config.TRAIN.POSE_LR = 0.0002 # 0.001
config.TRAIN.DIS_LR = 0.0002
config.TRAIN.TEMP_LR = 0.0002
config.TRAIN.WD = 0.0001
config.TRAIN.NUM_CRITICS = 3
config.TRAIN.NUM_CRITICS_TEMP = 3
config.TRAIN.NUM_EPOCHS = 200
config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.BOUND_AZIM = 3.1415926 # np.pi 
config.TRAIN.BOUND_ELEV = 3.1415926 / 9 # np.pi / 9

config.TRAIN.PRETRAINED = False 
# need to be updated
config.TRAIN.MODEL_PATH = ''

config.VIS = edict()
config.VIS.SCALE_MID_NUM_SEQ = 3

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.FIX = edict()
config.FIX.FIX_BONE_LOSS = False 
config.FIX.FIX_TRAJ = False
config.FIX.FIX_TRAJ_BY_ROT = False

def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
    for k, v in exp_config.items():
        if k in config:
            if isinstance(v, dict):
                _update_dict(k, v)
            else:
                config[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))
    assert not config.TRAIN.USE_TEMP or not config.TRAIN.USE_NEW_TEMP, "Don't use both temporal methods"
    assert config.DATA.NUM_NEIGHBOUR_TUPLES <= config.DATA.NUM_NEIGHBOUR_FRAMES * (config.DATA.NUM_NEIGHBOUR_FRAMES - 1) // 2

def update_dir(model_dir, log_dir, data_dir, debug):
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        config.OUTPUT_DIR = model_dir 
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    config.LOG_DIR = log_dir
    config.DATA_DIR = data_dir 
    if config.DATA.DATASET_NAME == 'surreal': 
        config.DATA.TRAIN_PATH = osp.join(config.DATA_DIR, "surreal_train_pred3.h5")
        config.DATA.VALID_PATH = osp.join(config.DATA_DIR, "surreal_valid_pred3.h5")
    elif config.DATA.DATASET_NAME == "h36m":
        # config.DATA.TRAIN_PATH = osp.join(config.DATA_DIR, "../../unsupervised_mesh/data/h36m_train_pred_3d_mesh.h5" if not debug else "debug_train.h5")
        # config.DATA.VALID_PATH = osp.join(config.DATA_DIR, "../../unsupervised_mesh/data/h36m_valid_pred_3d_mesh.h5" if not debug else "debug_valid.h5")
        config.DATA.TRAIN_PATH = osp.join(config.DATA_DIR, "h36m_train_pred3.h5" if not debug else "debug_train.h5")
        config.DATA.VALID_PATH = osp.join(config.DATA_DIR, "h36m_valid_pred3.h5" if not debug else "debug_valid.h5")
    elif config.DATA.DATASET_NAME == 'mpi': 
        config.DATA.TRAIN_PATH = osp.join(config.DATA_DIR, "mpi_train_pred3.h5")
        config.DATA.VALID_PATH = osp.join(config.DATA_DIR, "mpi_valid_pred3.h5")
    config.TRAIN.MODEL_PATH = osp.join(config.OUTPUT_DIR, "model.pkl")
