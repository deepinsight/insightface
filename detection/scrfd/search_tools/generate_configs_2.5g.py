
import os
import os.path as osp
import io
import numpy as np
import argparse
import datetime
import importlib
import configparser
from tqdm import tqdm
from mmdet.models import build_detector

import torch
import autotorch as at
from mmcv import Config

from mmdet.models import build_detector

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


@at.obj(
    block=at.Choice('BasicBlock', 'Bottleneck'),
    base_channels=at.Int(8, 64),
    stage_blocks=at.List(
        at.Int(1,10),
        at.Int(1,10),
        at.Int(1,10),
        at.Int(1,10),
        ),
    stage_planes_ratio=at.List(
        at.Real(1.0,4.0),
        at.Real(1.0,4.0),
        at.Real(1.0,4.0),
        ),
)
class GenConfigBackbone:
    def __init__(self, **kwargs):
        d = {}
        d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        self.m = 1.0

    def stage_blocks_multi(self, m):
        self.m = m

    def merge_cfg(self, det_cfg):

        base_channels = max(8, int(self.base_channels*self.m)//8 * 8)
        stage_planes = [base_channels]
        for ratio in self.stage_planes_ratio:
            planes = int(stage_planes[-1] * ratio) //8 * 8
            stage_planes.append(planes)
        stage_blocks = [max(1, int(x*self.m)) for x in self.stage_blocks]
        #print('Blocks:', stage_blocks)
        #print('Planes:', stage_planes)
        block_cfg=dict(block=self.block, stage_blocks=tuple(stage_blocks), stage_planes=stage_planes)
        det_cfg['model']['backbone']['block_cfg'] = block_cfg
        det_cfg['model']['backbone']['base_channels'] = base_channels
        neck_in_planes = stage_planes if self.block=='BasicBlock' else [4*x for x in stage_planes]
        det_cfg['model']['neck']['in_channels'] = neck_in_planes
        return det_cfg

@at.obj(
    stage_blocks_ratio=at.Real(0.5, 3.0),
    base_channels_ratio=at.Real(0.5, 3.0),
    fpn_channel=at.Int(8,128),
    head_channel=at.Int(8,256),
    head_stack=at.Int(1,4),
)
class GenConfigAll:
    def __init__(self, **kwargs):
        d = {}
        d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        self.m = 1

    def merge_cfg(self, det_cfg):
        block_cfg = det_cfg['model']['backbone']['block_cfg']
        stage_blocks = tuple([int(np.round(x*self.stage_blocks_ratio)) for x in block_cfg['stage_blocks']])
        block_cfg['stage_blocks'] = stage_blocks
        stage_planes = [int(np.round(x*self.base_channels_ratio))//8*8 for x in block_cfg['stage_planes']]
        block_cfg['stage_planes'] = stage_planes
        det_cfg['model']['backbone']['block_cfg'] = block_cfg
        det_cfg['model']['backbone']['base_channels'] = stage_planes[0]
        neck_in_planes = stage_planes if block_cfg['block']=='BasicBlock' else [4*x for x in stage_planes]
        det_cfg['model']['neck']['in_channels'] = neck_in_planes

        fpn_channel = self.fpn_channel//8*8
        head_channel = self.head_channel//8*8
        head_stack = self.head_stack
        det_cfg['model']['neck']['out_channels'] = fpn_channel
        det_cfg['model']['bbox_head']['in_channels'] = fpn_channel
        det_cfg['model']['bbox_head']['feat_channels'] = head_channel
        det_cfg['model']['bbox_head']['stacked_convs'] = head_stack
        gn_num_groups = 8
        for _gn_num_groups in [8, 16, 32, 64]:
            if head_channel%_gn_num_groups!=0:
                break
            gn_num_groups = _gn_num_groups
        det_cfg['model']['bbox_head']['norm_cfg']['num_groups'] = gn_num_groups
        return det_cfg

def get_args():
    parser = argparse.ArgumentParser(description='Auto-SCRFD')
    # config files
    parser.add_argument('--group', type=str, default='configs/scrfdgen2.5g', help='configs work dir')
    parser.add_argument('--template', type=int, default=0, help='template config index')
    parser.add_argument('--gflops', type=float, default=2.5, help='expected flops')
    parser.add_argument('--mode', type=int, default=1, help='generation mode, 1 for searching backbone, 2 for search all')
    # target flops
    parser.add_argument('--eps', type=float, default=2e-2,
                         help='eps for expected flops')
    # num configs
    parser.add_argument('--num-configs', type=int, default=64, help='num of expected configs')
    parser = parser

    args = parser.parse_args()
    return args


def is_config_valid(cfg, target_flops, input_shape, eps):
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=False, as_strings=False)
    print('FLOPs:', flops/1e9)
    return flops <= (1. + eps) * target_flops and \
        flops >= (1. - eps) * target_flops

def get_flops(cfg, input_shape):
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    #if torch.cuda.is_available():
    #    model.cuda()
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    buf = io.StringIO()
    all_flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=True, as_strings=False, ost=buf)
    buf = buf.getvalue()
    #print(buf)
    lines = buf.split("\n")
    names = ['(stem)', '(layer1)', '(layer2)', '(layer3)', '(layer4)', '(neck)', '(bbox_head)']
    name_ptr = 0
    line_num = 0
    _flops = []
    while name_ptr<len(names):
        line = lines[line_num].strip()
        name = names[name_ptr]
        if line.startswith(name):
            flops = float(lines[line_num+1].split(',')[2].strip().split(' ')[0])
            _flops.append(flops)
            name_ptr+=1
        line_num+=1

    backbone_flops = np.array(_flops[:-2], dtype=np.float32)
    neck_flops = _flops[-2]
    head_flops = _flops[-1]

    return all_flops/1e9, backbone_flops, neck_flops, head_flops

def is_flops_valid(flops, target_flops, eps):
    return flops <= (1. + eps) * target_flops and \
        flops >= (1. - eps) * target_flops

def main():
    args = get_args()
    print(datetime.datetime.now())

    input_shape = (3,480,640)
    runtime_input_shape = input_shape
    flops_mult = 1.0

    assert osp.exists(args.group)
    group_name = args.group.split('/')[-1]
    assert len(group_name)>0
    input_template = osp.join(args.group, "%s_%d.py"%(group_name, args.template))
    assert osp.exists(input_template)
    write_index = args.template+1
    while True:
        output_cfg = osp.join(args.group, "%s_%d.py"%(group_name, write_index))
        if not osp.exists(output_cfg):
            break
        write_index+=1
    print('write-index from:', write_index)

    if args.mode==1:
        gen = GenConfigBackbone()
    elif args.mode==2:
        gen = GenConfigAll()
        det_cfg = Config.fromfile(input_template)
        _, template_backbone_flops, _, _= get_flops(det_cfg, runtime_input_shape)
        template_backbone_ratios = list(map(lambda x:x/template_backbone_flops[0], template_backbone_flops))
        print('template_backbone_ratios:', template_backbone_ratios)



    pp = 0
    write_count = 0
    while write_count < args.num_configs:
        pp+=1
        det_cfg = Config.fromfile(input_template)
        config = gen.rand
        det_cfg = config.merge_cfg(det_cfg)
        all_flops, backbone_flops, neck_flops, head_flops = get_flops(det_cfg, runtime_input_shape)
        assert len(backbone_flops)==5
        all_flops *= flops_mult
        backbone_flops *= flops_mult
        neck_flops *= flops_mult
        head_flops *= flops_mult
        is_valid = True
        if pp%10==0:
            print(pp, all_flops, backbone_flops, neck_flops, head_flops, datetime.datetime.now())
        if args.mode==2:
            backbone_ratios = list(map(lambda x:x/backbone_flops[0], backbone_flops))
            #if head_flops*0.8<neck_flops:
            #    continue
            for i in range(1,5):
                if not is_flops_valid(template_backbone_ratios[i], backbone_ratios[i], args.eps*5):
                    is_valid = False
                    break
        if not is_valid:
            continue
        #if args.mode==1:
        #    if np.argmax(backbone_flops)!=1:
        #        continue
        #    if np.mean(backbone_flops[1:3])*0.8<np.mean(backbone_flops[-2:]):
        #        continue
        if not is_flops_valid(all_flops, args.gflops, args.eps):
            continue

        output_cfg_file = osp.join(args.group, "%s_%d.py"%(group_name, write_index))
        det_cfg.dump(output_cfg_file)
        print('SUCC', write_index, all_flops, backbone_flops, neck_flops, head_flops, datetime.datetime.now())
        write_index += 1
        write_count += 1

if __name__ == '__main__':
    main()

