import os
import time
import timm
import glob
import numpy as np
import os.path as osp

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from .iresnet import get_model as arcface_get_model


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()



        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(map_hidden_dim, map_hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(map_hidden_dim, map_hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        return torch.sin(freq * x + phase_shift)

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent).unsqueeze(2).unsqueeze(3) #B, 2*c, 1, 1
        gamma, beta = style.chunk(2, 1)
        x = gamma * x + beta
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type='reflect', activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out

class OneNetwork(nn.Module):
    def __init__(self, cfg):
        super(OneNetwork, self).__init__()
        self.num_verts = cfg.num_verts
        self.input_size = cfg.input_size
        self.use_eyes = cfg.eyes is not None
        kwargs = {}
        num_classes = self.num_verts*5
        if cfg.task==1:
            num_classes = self.num_verts*3
        elif cfg.task==2:
            num_classes = 6
        elif cfg.task==3:
            num_classes = self.num_verts*2
        eye_num_classes = 481*2*5
        #if use_eyes:
        #    num_classes += 481*2*5
        if cfg.network.startswith('resnet'):
            kwargs['base_width'] = int(64*cfg.width_mult)
        p_num_classes = num_classes
        if cfg.no_gap:
            p_num_classes = 0
            kwargs['global_pool'] = None
        elif cfg.use_arcface:
            p_num_classes = 0
            kwargs['global_pool'] = None
        elif self.use_eyes:
            p_num_classes = 0
            #kwargs['global_pool'] = None
        if cfg.network=='resnet_jmlr':
            from .resnet import resnet_jmlr
            self.net = resnet_jmlr(num_classes = p_num_classes, **kwargs)
            if self.use_eyes:
                input_dim = 512 #resnet34d
                self.nete = resnet_jmlr(num_classes = p_num_classes, **kwargs)
                self.fc = nn.Linear(input_dim*2, num_classes+eye_num_classes)
        else:
            self.net = timm.create_model(cfg.network, num_classes = p_num_classes, **kwargs)

        if cfg.no_gap:
            in_channel = self.net.num_features
            feat_hw = (self.input_size//32)**2
            mid_channel = 128
            self.no_gap_output = nn.Sequential(*[
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, mid_channel, 1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Flatten(1),
                nn.Linear(mid_channel*feat_hw, num_classes)])

        self.no_gap = cfg.no_gap
        self.use_arcface = cfg.use_arcface
        if self.use_arcface:
            self.neta = arcface_get_model(cfg.arcface_model, input_size=cfg.arcface_input_size)
            self.neta.load_state_dict(torch.load(cfg.arcface_ckpt, map_location=torch.device('cpu')))
            self.neta.eval()
            self.neta.requires_grad_(False)
            input_dim = 512 #resnet34d
            z_dim = 512 #arcface_dim
            hidden_dim = 256
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten(1)
            mlp_act = nn.LeakyReLU

            self.mlp = nn.Sequential(*[
                nn.Linear(z_dim, hidden_dim),
                mlp_act(),
                nn.Linear(hidden_dim, hidden_dim),
                mlp_act(),
                nn.Linear(hidden_dim, hidden_dim),
                mlp_act(),
                nn.Linear(hidden_dim, hidden_dim),
                mlp_act(),
                nn.Linear(hidden_dim, hidden_dim),
                mlp_act(),
                nn.Linear(hidden_dim, hidden_dim),
                mlp_act(),
                nn.Linear(hidden_dim, input_dim),
                ])
            style_blocks = []
            for i in range(3):
                style_blocks += [ResnetBlock_Adain(input_dim, latent_size=z_dim)]
            self.style_blocks = nn.Sequential(*style_blocks)
            self.branch2d = nn.Sequential(*[
                nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(input_dim),
                nn.ReLU(),
                ])
            self.fc = nn.Linear(input_dim*2, num_classes)


    def forward(self, x):
        if self.use_arcface:
            conv_feat = self.net.forward_features(x)
            #input = self.flatten(self.pool(conv_feat))
            xa = F.interpolate(x, [144, 144], mode='bilinear', align_corners=False)
            xa = xa[:,:,8:120,16:128]
            z = self.neta(xa)
            z = self.mlp(z)

            c = conv_feat
            for i in range(len(self.style_blocks)):
                c = self.style_blocks[i](c, z)
            feat3 = c
            feat2 = self.branch2d(conv_feat)
            conv_feat = torch.cat([feat3, feat2], dim=1)
            feat = self.flatten(self.pool(conv_feat))
            pred = self.fc(feat)

        elif self.no_gap:
            y = self.net.forward_features(x)
            pred = self.no_gap_output(y)
        else:
            pred = self.net(x)
            if self.use_eyes:
                eye_w1 = x.shape[3]//8
                eye_w2 = x.shape[3] - wa
                hstep = x.shape[2]//8
                eye_h1 = hstep*2
                eye_h2 = hstep*4
                x_eye = x[:,:,eye_h1:eye_h2,eye_w1:eye_w2]
                feate = self.nete(x_eye)
                #print(pred.shape, feate.shape)
                feat = torch.cat((pred,feate), 1)
                pred = self.fc(feat)
        return pred

def get_network(cfg):
    if cfg.use_onenetwork:
        net = OneNetwork(cfg)
    else:
        net = timm.create_model(cfg.network, num_classes = 1220*5)
    return net


