import torch
from torch import nn
import torch.nn.functional as F
import logging

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, eps=1e-5, dropblock=0.0):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=eps)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=eps)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=eps)
        self.downsample = downsample
        self.stride = stride
        self.dbs = None
        if dropblock>0.0:
            import timm
            from timm.layers import DropBlock2d
            self.dbs = [DropBlock2d(dropblock, 7), DropBlock2d(dropblock, 7), DropBlock2d(dropblock, 7)]

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        if self.dbs is not None:
            out = self.dbs[0](out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        if self.dbs is not None:
            out = self.dbs[1](out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.dbs is not None:
            out = self.dbs[2](out)
        return out


class IResNet(nn.Module):
    def __init__(self,
                 block, layers, dropout=0.0, num_features=512, input_size=112, zero_init_residual=False,
                 stem_type='', dropblock = 0.0, kaiming_init=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=0):
        super(IResNet, self).__init__()
        self.input_size = input_size
        assert self.input_size%16==0
        fc_scale = self.input_size // 16
        self.fc_scale = fc_scale*fc_scale
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.norm_layer = nn.BatchNorm2d
        self.act_layer = nn.PReLU
        self.eps = 1e-5
        if kaiming_init:
            self.eps = 2e-5
        self.stem_type = stem_type
        self.dropblock = dropblock
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if stem_type!='D':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            stem_width = self.inplanes // 2
            stem_chs = [stem_width, stem_width]
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(3, stem_chs[0], 3, stride=1, padding=1, bias=False),
                self.norm_layer(stem_chs[0], eps=self.eps),
                self.act_layer(stem_chs[0]),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                self.norm_layer(stem_chs[1], eps=self.eps),
                self.act_layer(stem_chs[1]),
                nn.Conv2d(stem_chs[1], self.inplanes, 3, stride=1, padding=1, bias=False)])
        logging.info("iresnet, input_size: %d, fc_scale: %d, dropout: %.2f, stem_type: %s, fp16: %d"%(self.input_size, self.fc_scale, dropout, stem_type, self.fp16))
        logging.info("iresnet, eps: %.6f, dropblock: %.3f, kaiming_init: %d"%(self.eps, self.dropblock, kaiming_init))
        #self.conv1.requires_grad = False
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=self.eps)
        #self.bn1.requires_grad = False
        self.prelu = nn.PReLU(self.inplanes)
        #self.prelu.requires_grad = False
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        #self.layer1.requires_grad = False
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        #self.layer2.requires_grad = False
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       dropblock=self.dropblock)
        #self.layer3.requires_grad = False
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       dropblock=self.dropblock)
        #self.layer4.requires_grad = False
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=self.eps)
        #self.bn2.requires_grad = False
        if dropout>0.0:
            self.dropout = nn.Dropout(p=dropout, inplace=True)
        else:
            self.dropout = None
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        #self.fc.requires_grad = False
        self.features = nn.BatchNorm1d(num_features, eps=self.eps)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform_(m.weight.data)
        #        if m.bias is not None:
        #            m.bias.data.zero_()
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()
        #    elif isinstance(m, nn.BatchNorm1d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()
        #    elif isinstance(m, nn.Linear):
        #        nn.init.xavier_uniform_(m.weight.data)
        #        if m.bias is not None:
        #            m.bias.data.zero_()
        #nn.init.constant_(self.features.weight, 1.0)
        #self.features.weight.requires_grad = False

        #for m in self.modules():
        #    if kaiming_init:
        #        if isinstance(m, (nn.Conv2d, nn.Linear)):
        #            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #            if m.bias is not None:
        #                nn.init.constant_(m.bias, 0)
        #    else:
        #        if isinstance(m, (nn.Conv2d, nn.Linear)):
        #            nn.init.normal_(m.weight, 0, 0.1)
        #            if m.bias is not None:
        #                nn.init.constant_(m.bias, 0)
        #    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, dropblock=0.0):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.stem_type!='D':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion, eps=self.eps),
                )
            else:
				#avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
                avg_stride = stride
                pool = nn.AvgPool2d(2, avg_stride, ceil_mode=True, count_include_pad=False)
                downsample = nn.Sequential(*[
                    pool,
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    nn.BatchNorm2d(planes * block.expansion, eps=self.eps),
                    ])
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, self.eps, dropblock))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      eps=self.eps,
                      dropblock=dropblock))

        return nn.Sequential(*layers)

    def forward(self, x):
        #if self.input_size!=112:
        #    x = F.interpolate(x, [self.input_size, self.input_size], mode='bilinear', align_corners=False)
        is_fp16 = self.fp16>0
        with torch.cuda.amp.autocast(is_fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            if self.fp16<3:
                x = self.layer4(x)
                x = self.bn2(x)
                x = torch.flatten(x, 1)
                if self.dropout is not None:
                    x = self.dropout(x)
        if is_fp16:
            x = x.float()
        if self.fp16>=3:
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            if self.dropout is not None:
                x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)

def iresnet120(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet120', IBasicBlock, [3, 16, 37, 3], pretrained,
                    progress, **kwargs)

def iresnet160(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet160', IBasicBlock, [3, 16, 56, 3], pretrained,
                    progress, **kwargs)

def iresnet180(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet180', IBasicBlock, [3, 20, 63, 3], pretrained,
                    progress, **kwargs)

def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)

def iresnet247(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet247', IBasicBlock, [3, 36, 80, 4], pretrained,
                    progress, **kwargs)

def iresnet269(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet269', IBasicBlock, [4, 46, 80, 4], pretrained,
                    progress, **kwargs)

def iresnet300(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet300', IBasicBlock, [4, 46, 95, 4], pretrained,
                    progress, **kwargs)


def get_model(name, **kwargs):
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)
    else:
        raise ValueError()

