import numpy as np
from easydict import EasyDict as edict

config = edict()

# network related params
config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
config.PIXEL_STDS = np.array([1.0, 1.0, 1.0])
config.PIXEL_SCALE = 1.0
config.IMAGE_STRIDE = 0

# dataset related params
config.NUM_CLASSES = 2
config.PRE_SCALES = [(1200, 1600)
                     ]  # first is scale (the shorter side); second is max size
config.SCALES = [(640, 640)
                 ]  # first is scale (the shorter side); second is max size
#config.SCALES = [(800, 800)]  # first is scale (the shorter side); second is max size
config.ORIGIN_SCALE = False

_ratio = (1., )

RAC_SSH = {
    '32': {
        'SCALES': (32, 16),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
    '16': {
        'SCALES': (8, 4),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
    '8': {
        'SCALES': (2, 1),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
}

_ratio = (1., 1.5)
RAC_SSH2 = {
    '32': {
        'SCALES': (32, 16),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
    '16': {
        'SCALES': (8, 4),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
    '8': {
        'SCALES': (2, 1),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
}

_ratio = (1., 1.5)
RAC_SSH3 = {
    '32': {
        'SCALES': (32, 16),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
    '16': {
        'SCALES': (8, 4),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
    '8': {
        'SCALES': (2, 1),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
    '4': {
        'SCALES': (2, 1),
        'BASE_SIZE': 16,
        'RATIOS': _ratio,
        'ALLOWED_BORDER': 9999
    },
}

RAC_RETINA = {}
_ratios = (1.0, )
_ass = 2.0**(1.0 / 3)
_basescale = 1.0
for _stride in [4, 8, 16, 32, 64]:
    key = str(_stride)
    value = {'BASE_SIZE': 16, 'RATIOS': _ratios, 'ALLOWED_BORDER': 9999}
    scales = []
    for _ in range(3):
        scales.append(_basescale)
        _basescale *= _ass
    value['SCALES'] = tuple(scales)
    RAC_RETINA[key] = value

config.RPN_ANCHOR_CFG = RAC_SSH  #default

config.NET_MODE = 2
config.HEAD_MODULE = 'SSH'
#config.HEAD_MODULE = 'RF'
config.LR_MODE = 0
config.LANDMARK_LR_MULT = 2.0
config.HEAD_FILTER_NUM = 256
config.CONTEXT_FILTER_RATIO = 1
config.max_feat_channel = 9999

config.USE_CROP = True
config.USE_FPN = True
config.USE_DCN = 0
config.FACE_LANDMARK = True
config.USE_OCCLUSION = False
config.USE_BLUR = False
config.MORE_SMALL_BOX = True

config.LAYER_FIX = False

config.CASCADE = 0
config.CASCADE_MODE = 1
#config.CASCADE_CLS_STRIDES = [16,8,4]
#config.CASCADE_BBOX_STRIDES = [64,32]
config.CASCADE_CLS_STRIDES = [64, 32, 16, 8, 4]
config.CASCADE_BBOX_STRIDES = [64, 32, 16, 8, 4]
#config.CASCADE_BBOX_STRIDES = [64,32,16,8]

config.HEAD_BOX = False
config.DENSE_ANCHOR = False
config.USE_MAXOUT = 0
config.SHARE_WEIGHT_BBOX = False
config.SHARE_WEIGHT_LANDMARK = False

config.RANDOM_FEAT_STRIDE = False
config.NUM_CPU = 4
config.MIXUP = 0.0
config.USE_3D = False

#config.BBOX_MASK_THRESH = 0
config.COLOR_MODE = 2
config.COLOR_JITTERING = 0.125
#config.COLOR_JITTERING = 0
#config.COLOR_JITTERING = 0.2

config.TRAIN = edict()

config.TRAIN.IMAGE_ALIGN = 0
config.TRAIN.MIN_BOX_SIZE = 0
config.BBOX_MASK_THRESH = config.TRAIN.MIN_BOX_SIZE
# R-CNN and RPN
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 8
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = True
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = False

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_ENABLE_OHEM = 2
config.TRAIN.OHEM_MODE = 1
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.25
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.5
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
if config.CASCADE > 0:
    config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.CASCADE_OVERLAP = [0.4, 0.5]
config.TRAIN.RPN_CLOBBER_POSITIVES = False
config.TRAIN.RPN_FORCE_POSITIVE = False
# rpn bounding box regression params
config.TRAIN.BBOX_STDS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.LANDMARK_STD = 1.0

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.3
config.TEST.RPN_PRE_NMS_TOP_N = 1000
config.TEST.RPN_POST_NMS_TOP_N = 3000
#config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
#config.TEST.RPN_MIN_SIZE = [0,0,0]

# RCNN nms
config.TEST.NMS = 0.3

config.TEST.SCORE_THRESH = 0.05
config.TEST.IOU_THRESH = 0.5

# network settings
network = edict()

network.ssh = edict()

network.mnet = edict()
#network.mnet.pretrained = 'model/mnasnet'
#network.mnet.pretrained = 'model/mobilenetv2_0_5'
#network.mnet.pretrained = 'model/mobilenet_0_5'
#network.mnet.MULTIPLIER = 0.5
#network.mnet.pretrained = 'model/mobilenet_0_25'
#network.mnet.pretrained_epoch = 0
#network.mnet.PIXEL_MEANS = np.array([0.406, 0.456, 0.485])
#network.mnet.PIXEL_STDS = np.array([0.225, 0.224, 0.229])
#network.mnet.PIXEL_SCALE = 255.0
network.mnet.FIXED_PARAMS = ['^stage1', '^.*upsampling']
network.mnet.BATCH_IMAGES = 16
network.mnet.HEAD_FILTER_NUM = 64
network.mnet.CONTEXT_FILTER_RATIO = 1

network.mnet.PIXEL_MEANS = np.array([0.0, 0.0, 0.0])
network.mnet.PIXEL_STDS = np.array([1.0, 1.0, 1.0])
network.mnet.PIXEL_SCALE = 1.0
#network.mnet.pretrained = 'model/mobilenetfd_0_25' #78
#network.mnet.pretrained = 'model/mobilenetfd2' #75
network.mnet.pretrained = 'model/mobilenet025fd0'  #78
#network.mnet.pretrained = 'model/mobilenet025fd1' #75
#network.mnet.pretrained = 'model/mobilenet025fd2' #
network.mnet.pretrained_epoch = 0
network.mnet.max_feat_channel = 8888
network.mnet.COLOR_MODE = 1
network.mnet.USE_CROP = True
network.mnet.RPN_ANCHOR_CFG = RAC_SSH
network.mnet.LAYER_FIX = True
network.mnet.LANDMARK_LR_MULT = 2.5

network.resnet = edict()
#network.resnet.pretrained = 'model/ResNet50_v1d'
#network.resnet.pretrained = 'model/resnet-50'
network.resnet.pretrained = 'model/resnet-152'
#network.resnet.pretrained = 'model/senet154'
#network.resnet.pretrained = 'model/densenet161'
network.resnet.pretrained_epoch = 0
#network.mnet.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
#network.mnet.PIXEL_STDS = np.array([57.375, 57.12, 58.393])
#network.resnet.PIXEL_MEANS = np.array([0.406, 0.456, 0.485])
#network.resnet.PIXEL_STDS = np.array([0.225, 0.224, 0.229])
#network.resnet.PIXEL_SCALE = 255.0
network.resnet.lr_step = '1,2,3,4,5,55,68,80'
network.resnet.lr = 0.001
network.resnet.PIXEL_MEANS = np.array([0.0, 0.0, 0.0])
network.resnet.PIXEL_STDS = np.array([1.0, 1.0, 1.0])
network.resnet.PIXEL_SCALE = 1.0
network.resnet.FIXED_PARAMS = ['^stage1', '^.*upsampling']
network.resnet.BATCH_IMAGES = 8
network.resnet.HEAD_FILTER_NUM = 256
network.resnet.CONTEXT_FILTER_RATIO = 1
network.resnet.USE_DCN = 2
network.resnet.RPN_BATCH_SIZE = 256
network.resnet.RPN_ANCHOR_CFG = RAC_RETINA

network.resnet.USE_DCN = 0
network.resnet.pretrained = 'model/resnet-50'
network.resnet.RPN_ANCHOR_CFG = RAC_SSH

# dataset settings
dataset = edict()

dataset.widerface = edict()
dataset.widerface.dataset = 'widerface'
dataset.widerface.image_set = 'train'
dataset.widerface.test_image_set = 'val'
dataset.widerface.root_path = 'data'
dataset.widerface.dataset_path = 'data/widerface'
dataset.widerface.NUM_CLASSES = 2

dataset.retinaface = edict()
dataset.retinaface.dataset = 'retinaface'
dataset.retinaface.image_set = 'train'
dataset.retinaface.test_image_set = 'val'
dataset.retinaface.root_path = 'data'
dataset.retinaface.dataset_path = 'data/retinaface'
dataset.retinaface.NUM_CLASSES = 2

# default settings
default = edict()

config.FIXED_PARAMS = ['^conv1', '^conv2', '^conv3', '^.*upsampling']
#config.FIXED_PARAMS = ['^.*upsampling']
#config.FIXED_PARAMS = ['^conv1', '^conv2', '^conv3']
#config.FIXED_PARAMS = ['^conv0', '^stage1', 'gamma', 'beta']  #for resnet

# default network
default.network = 'resnet'
default.pretrained = 'model/resnet-152'
#default.network = 'resnetssh'
default.pretrained_epoch = 0
# default dataset
default.dataset = 'retinaface'
default.image_set = 'train'
default.test_image_set = 'val'
default.root_path = 'data'
default.dataset_path = 'data/retinaface'
# default training
default.frequent = 20
default.kvstore = 'device'
# default e2e
default.prefix = 'model/retinaface'
default.end_epoch = 10000
default.lr_step = '55,68,80'
default.lr = 0.01
default.wd = 0.0005


def generate_config(_network, _dataset):
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
        if k in config.TRAIN:
            config.TRAIN[k] = v
    for k, v in dataset[_dataset].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
        if k in config.TRAIN:
            config.TRAIN[k] = v
    config.network = _network
    config.dataset = _dataset
    config.RPN_FEAT_STRIDE = []
    num_anchors = []
    for k in config.RPN_ANCHOR_CFG:
        config.RPN_FEAT_STRIDE.append(int(k))
        _num_anchors = len(config.RPN_ANCHOR_CFG[k]['SCALES']) * len(
            config.RPN_ANCHOR_CFG[k]['RATIOS'])
        if config.DENSE_ANCHOR:
            _num_anchors *= 2
        config.RPN_ANCHOR_CFG[k]['NUM_ANCHORS'] = _num_anchors
        num_anchors.append(_num_anchors)
    config.RPN_FEAT_STRIDE = sorted(config.RPN_FEAT_STRIDE, reverse=True)
    for j in range(1, len(num_anchors)):
        assert num_anchors[0] == num_anchors[j]
    config.NUM_ANCHORS = num_anchors[0]
