import argparse
import os
import os.path as osp
import pickle
import numpy as np
import datetime
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.core.evaluation import wider_evaluation, get_widerface_gts
#from torch.utils import mkldnn as mkldnn_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[480, 640],
        help='input image size')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()


    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    pipelines = cfg.data.test.pipeline
    for pipeline in pipelines:
        if pipeline.type=='MultiScaleFlipAug':
            #pipeline.img_scale = (640, 640)
            pipeline.img_scale = None
            pipeline.scale_factor = 1.0
            transforms = pipeline.transforms
            for transform in transforms:
                if transform.type=='Pad':
                    #transform.size = pipeline.img_scale
                    transform.size = None
                    transform.size_divisor = 1
    #print(cfg.data.test.pipeline)
    distributed = False

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    device = torch.device("cpu" if args.cpu else "cuda")

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = model.to(device)

    model.eval()
    dataset = data_loader.dataset
    for i, data in enumerate(data_loader):
        img = data['img'][0]
        #print(img.shape)
        img = img[:,:,:args.shape[0],:args.shape[1]]
        img = img.to(device)
        with torch.no_grad():
            ta = datetime.datetime.now()
            result = model.feature_test(img)
            tb = datetime.datetime.now()
            print('cost:', (tb-ta).total_seconds())
            





if __name__ == '__main__':
    main()
