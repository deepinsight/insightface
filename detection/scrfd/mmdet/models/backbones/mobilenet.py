from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from mmdet.utils import get_root_logger
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES

@BACKBONES.register_module()
class MobileNetV1(nn.Module):
    def __init__(self,
                 in_channels=3,
                 #base_channels=32,
                 block_cfg = None,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3)):
        super(MobileNetV1, self).__init__()
        self.out_indices = out_indices

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        if block_cfg is None:
            stage_planes = [8, 16, 32, 64, 128, 256]  #0.25 default
            stage_blocks = [2,4,4,2]
        else:
            stage_planes = block_cfg['stage_planes']
            stage_blocks = block_cfg['stage_blocks']
        assert len(stage_planes)==6
        assert len(stage_blocks)==4
        self.stem = nn.Sequential(
                conv_bn(3, stage_planes[0], 2),
                conv_dw(stage_planes[0], stage_planes[1], 1),
                )
        self.stage_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            _layers = []
            for n in range(num_blocks):
                if n==0:
                    _layer = conv_dw(stage_planes[i+1], stage_planes[i+2], 2)
                else:
                    _layer = conv_dw(stage_planes[i+2], stage_planes[i+2], 1)
                _layers.append(_layer)

            _block = nn.Sequential(*_layers)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, _block)
            self.stage_layers.append(layer_name)

        #bc = base_channels
        #self.stages = nn.ModuleDict()
        #self.stage0 = nn.Sequential(
        #    conv_bn(3, bc, 2),
        #    conv_dw(bc, bc*2, 1),
        #    conv_dw(bc*2, bc*4, 2),
        #    conv_dw(bc*4, bc*4, 1),
        #)
        #self.stage1 = nn.Sequential(
        #    conv_dw(bc*4, bc*8, 2),
        #    conv_dw(bc*8, bc*8, 1),

        #    conv_dw(bc*8, bc*8, 1),
        #    conv_dw(bc*8, bc*8, 1),
        #)
        #self.stage2 = nn.Sequential(
        #    conv_dw(bc*8, bc*16, 2),
        #    conv_dw(bc*16, bc*16, 1),
        #    conv_dw(bc*16, bc*16, 1),
        #    conv_dw(bc*16, bc*16, 1),
        #    #conv_dw(bc*16, bc*16, 1),
        #    #conv_dw(bc*16, bc*16, 1),
        #)
        #self.stage3 = nn.Sequential(
        #    conv_dw(bc*16, bc*32, 2),
        #    conv_dw(bc*32, bc*32, 1),
        #)
        #self.stages = [self.stage0, self.stage1, self.stage2, self.stage3]



    def forward(self, x):
        output = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.stage_layers):
            stage_layer = getattr(self, layer_name)
            x = stage_layer(x)
            if i in self.out_indices:
                output.append(x)

        return tuple(output)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
