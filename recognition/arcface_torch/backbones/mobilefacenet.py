'''
Adapted from https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/mobilefacenet.py
Original author cavalleria
'''

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import namedtuple
import math


##################################  Common #############################################################
def round_channels(channels, divisor=8):
    """
    Round weighted channel number (make divisible operation).
    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.
    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(
        int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


class ECA_Layer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1,
                              1,
                              kernel_size=k_size,
                              padding=(k_size - 1) // 2,
                              bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1,
                                              -2)).transpose(-1,
                                                             -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 round_mid=False,
                 use_conv=True,
                 mid_activation=(lambda: nn.ReLU(inplace=True)),
                 out_activation=(lambda: nn.Sigmoid())):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        mid_channels = channels // reduction if not round_mid else round_channels(
            float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2d(in_channels=channels,
                                   out_channels=mid_channels,
                                   kernel_size=1,
                                   stride=1,
                                   groups=1,
                                   bias=True)
        else:
            self.fc1 = nn.Linear(in_features=channels,
                                 out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        if use_conv:
            self.conv2 = nn.Conv2d(in_channels=mid_channels,
                                   out_channels=channels,
                                   kernel_size=1,
                                   stride=1,
                                   groups=1,
                                   bias=True)
        else:
            self.fc2 = nn.Linear(in_features=mid_channels,
                                 out_features=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x


##################################  Original Arcface Model #############################################################
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


##################################  MobileFaceNet #############################################################
class Conv_block(Module):
    def __init__(self,
                 in_c,
                 out_c,
                 kernel=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c,
                           out_channels=out_c,
                           kernel_size=kernel,
                           groups=groups,
                           stride=stride,
                           padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self,
                 in_c,
                 out_c,
                 kernel=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c,
                           out_channels=out_c,
                           kernel_size=kernel,
                           groups=groups,
                           stride=stride,
                           padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(self,
                 in_c,
                 out_c,
                 attention,
                 residual=False,
                 kernel=(3, 3),
                 stride=(2, 2),
                 padding=(1, 1),
                 groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c,
                               out_c=groups,
                               kernel=(1, 1),
                               padding=(0, 0),
                               stride=(1, 1))
        self.conv_dw = Conv_block(groups,
                                  groups,
                                  groups=groups,
                                  kernel=kernel,
                                  padding=padding,
                                  stride=stride)
        self.project = Linear_block(groups,
                                    out_c,
                                    kernel=(1, 1),
                                    padding=(0, 0),
                                    stride=(1, 1))
        self.attention = attention
        if self.attention == 'eca':
            self.attention_layer = ECA_Layer(out_c)
        elif self.attention == 'se':
            self.attention_layer = SEBlock(out_c)
        # elif self.attention == 'cbam':
        #     self.attention_layer = CbamBlock(out_c)
        # elif self.attention == 'gct':
        #     self.attention_layer = GCT(out_c)

        self.residual = residual

        self.attention = attention  #se, eca, cbam

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.attention != 'none':
            x = self.attention_layer(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self,
                 c,
                 attention,
                 num_block,
                 groups,
                 kernel=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c,
                           c,
                           attention,
                           residual=True,
                           kernel=kernel,
                           padding=padding,
                           stride=stride,
                           groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class GNAP(Module):
    def __init__(self, embedding_size):
        super(GNAP, self).__init__()
        assert embedding_size == 512
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = Linear_block(512,
                                      512,
                                      groups=512,
                                      kernel=(7, 7),
                                      stride=(1, 1),
                                      padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        #self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class MobileFaceNet(Module):
    def __init__(self,
                 input_size,
                 dropout=0,
                 fp16=False,
                 num_features=512,
                 output_name="GDC",
                 attention='none'):
        super(MobileFaceNet, self).__init__()
        assert output_name in ['GNAP', 'GDC']
        assert input_size[0] in [112]
        assert fp16 is False, "MobileFaceNet not support fp16 mode;)"

        self.conv1 = Conv_block(3,
                                64,
                                kernel=(3, 3),
                                stride=(2, 2),
                                padding=(1, 1))
        self.conv2_dw = Conv_block(64,
                                   64,
                                   kernel=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   groups=64)
        self.conv_23 = Depth_Wise(64,
                                  64,
                                  attention,
                                  kernel=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1),
                                  groups=128)
        self.conv_3 = Residual(64,
                               attention,
                               num_block=4,
                               groups=128,
                               kernel=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv_34 = Depth_Wise(64,
                                  128,
                                  attention,
                                  kernel=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1),
                                  groups=256)
        self.conv_4 = Residual(128,
                               attention,
                               num_block=6,
                               groups=256,
                               kernel=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv_45 = Depth_Wise(128,
                                  128,
                                  attention,
                                  kernel=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1),
                                  groups=512)
        self.conv_5 = Residual(128,
                               attention,
                               num_block=2,
                               groups=256,
                               kernel=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv_6_sep = Conv_block(128,
                                     512,
                                     kernel=(1, 1),
                                     stride=(1, 1),
                                     padding=(0, 0))
        if output_name == "GNAP":
            self.output_layer = GNAP(512)
        else:
            self.output_layer = GDC(num_features)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        conv_features = self.conv_6_sep(out)
        out = self.output_layer(conv_features)
        return out
