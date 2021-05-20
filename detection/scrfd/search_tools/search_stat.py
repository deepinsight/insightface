import os
import json
import os.path as osp
import io
import torch
import numpy as np
from mmcv import Config

from mmdet.models import build_detector
from mmcv.cnn import get_model_complexity_info

def get_flops(cfg, input_shape):
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
    buf = io.StringIO()
    all_flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=True, as_strings=False, ost=buf)
    buf = buf.getvalue()
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

    backbone_flops = _flops[:-2]
    neck_flops = _flops[-2]
    head_flops = _flops[-1]
    return all_flops/1e9, backbone_flops, neck_flops, head_flops
def get_stat(result_dir, group, prefix, idx):
    curr_dir = osp.join(result_dir, group, "%s_%d"%(prefix, idx))
    aps_file = osp.join(curr_dir, 'aps')
    aps = []
    if osp.exists(aps_file):
        with open(aps_file, 'r') as f:
            aps = [float(x) for x in f.readline().strip().split(',')]
    cfg_file = osp.join('configs', group, '%s_%d.py'%(prefix, idx))
    cfg = Config.fromfile(cfg_file)
    all_flops, backbone_flops, neck_flops, head_flops = get_flops(cfg, (3,480,640))
    
    return aps, all_flops, backbone_flops, neck_flops, head_flops

result_dir = './wouts'
group = 'scrfdgen2.5g'
prefix = group
idx_from = 0
idx_to = 320

outf = open(osp.join(result_dir, "%s.txt"%prefix), 'w')

for idx in range(idx_from, idx_to):
    aps, all_flops, backbone_flops, neck_flops, head_flops = get_stat(result_dir, group, prefix, idx)
    backbone_ratio = np.sum(backbone_flops) / all_flops
    neck_ratio = neck_flops / all_flops
    head_ratio = head_flops / all_flops
    print(idx, aps, all_flops, backbone_flops, backbone_ratio, neck_ratio, head_ratio)
    name = "%s_%d"%(prefix, idx)
    data = dict(name=name, backbone_flops=backbone_flops, neck_flops=neck_flops, head_flops=head_flops, all_flops=all_flops, aps=aps)
    data = json.dumps(data)
    outf.write(data)
    outf.write("\n")
outf.close()

