import os
import torch


def get_cur_epoch(cfg):
    list_file = os.listdir(cfg.output)
    cur_epoch = 0
    if cfg.resume:
        for each in list_file:
            if 'model_epoch_' in each:
                epoch = int(os.path.splitext(each)[0].split('_')[-1])
                if epoch > cur_epoch:
                    cur_epoch = epoch
    return cur_epoch


def load_backbone(cfg, backbone, start_epoch, rank):
    if cfg.resume:
        backbone_path = os.path.join(
            cfg.output, f'model_epoch_{start_epoch}.pt')
        print(backbone_path)
        if os.path.exists(backbone_path):
            backbone.load_state_dict(torch.load(backbone_path))


def load_partial_fc(cfg, module_partial_fc, rank):
    if cfg.resume:
        partial_fc_path = os.path.join(
            cfg.output, f'softmax_fc_gpu_{rank}.pt')
        print(partial_fc_path)
        if os.path.exists(partial_fc_path):
            module_partial_fc.load_state_dict(torch.load(partial_fc_path))
