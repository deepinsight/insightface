import numpy as np
from easydict import EasyDict as edict

config = edict()

#default training/dataset config
config.num_classes = 68
config.record_img_size = 384
config.base_scale = 256
config.input_img_size = 128
config.output_label_size = 64
config.label_xfirst = False
config.losstype = 'heatmap'
config.net_coherent = False
config.multiplier = 1.0

config.gaussian = 0

# network settings
network = edict()

network.hourglass = edict()
network.hourglass.net_coherent = False
network.hourglass.net_sta = 0
network.hourglass.net_n = 3
network.hourglass.net_dcn = 0
network.hourglass.net_stacks = 2
network.hourglass.net_block = 'resnet'
network.hourglass.net_binarize = False
network.hourglass.losstype = 'heatmap'

network.sdu = edict()
network.sdu.net_coherent = False
network.sdu.net_sta = 1
network.sdu.net_n = 3
network.sdu.net_dcn = 3
network.sdu.net_stacks = 2
network.sdu.net_block = 'cab'
network.sdu.net_binarize = False
network.sdu.losstype = 'heatmap'


# dataset settings
dataset = edict()

dataset.i2d = edict()
dataset.i2d.dataset = '2D'
dataset.i2d.landmark_type = '2d'
dataset.i2d.dataset_path = './data_2d'
dataset.i2d.num_classes = 68
dataset.i2d.record_img_size = 384
dataset.i2d.base_scale = 256
dataset.i2d.input_img_size = 128
dataset.i2d.output_label_size = 64
dataset.i2d.label_xfirst = False
dataset.i2d.val_targets = ['ibug', 'cofw_testset', '300W']

dataset.i3d = edict()
dataset.i3d.dataset = '3D'
dataset.i3d.landmark_type = '3d'
dataset.i3d.dataset_path = './data_3d'
dataset.i3d.num_classes = 68
dataset.i3d.record_img_size = 384
dataset.i3d.base_scale = 256
dataset.i3d.input_img_size = 128
dataset.i3d.output_label_size = 64
dataset.i3d.label_xfirst = False
dataset.i3d.val_targets = ['AFLW2000-3D']


# default settings
default = edict()

# default network
default.network = 'hourglass'
default.pretrained = ''
default.pretrained_epoch = 0
# default dataset
default.dataset = 'i2d'
default.frequent = 20
default.verbose = 200
default.kvstore = 'device'

default.prefix = 'model/A'
default.end_epoch = 10000
default.lr = 0.00025
default.wd = 0.0
default.per_batch_size = 20
default.lr_step = '16000,24000,30000'

def generate_config(_network, _dataset):
    for k, v in network[_network].items():
      config[k] = v
      default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      default[k] = v
    config.network = _network
    config.dataset = _dataset

