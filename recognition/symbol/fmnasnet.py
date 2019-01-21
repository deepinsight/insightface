import sys
import os
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

def Act():
    if config.net_act=='prelu':
      return nn.PReLU()
    else:
      return nn.Activation(config.net_act)

def ConvBlock(channels, kernel_size, strides, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, kernel_size, strides=strides, padding=1, use_bias=False),
            nn.BatchNorm(scale=True),
            Act()
            #nn.Activation('relu')
        )
    return out

def Conv1x1(channels, is_linear=False, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, 1, padding=0, use_bias=False),
            nn.BatchNorm(scale=True)
        )
        if not is_linear:
            #out.add(nn.Activation('relu'))
            out.add(Act())
    return out

def DWise(channels, strides, kernel_size=3, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, kernel_size, strides=strides, padding=kernel_size // 2, groups=channels, use_bias=False),
            nn.BatchNorm(scale=True),
            Act()
            #nn.Activation('relu')
        )
    return out

class SepCONV(nn.HybridBlock):
    def __init__(self, inp, output, kernel_size, depth_multiplier=1, with_bn=True, **kwargs):
        super(SepCONV, self).__init__(**kwargs)
        with self.name_scope():
            self.net = nn.HybridSequential()
            cn = int(inp*depth_multiplier)

            if output is None:
                self.net.add(
                    nn.Conv2D(in_channels=inp, channels=cn, groups=inp, kernel_size=kernel_size, strides=(1,1), padding=kernel_size // 2
                        , use_bias=not with_bn)
                )
            else:
                self.net.add(
                    nn.Conv2D(in_channels=inp, channels=cn, groups=inp, kernel_size=kernel_size, strides=(1,1), padding=kernel_size // 2
                        , use_bias=False),
                    nn.BatchNorm(),
                    Act(),
                    #nn.Activation('relu'),
                    nn.Conv2D(in_channels=cn, channels=output, kernel_size=(1,1), strides=(1,1)
                        , use_bias=not with_bn)
                )

            self.with_bn = with_bn
            self.act = Act()
            #self.act = nn.Activation('relu')
            if with_bn:
                self.bn = nn.BatchNorm()
    def hybrid_forward(self, F ,x):
        x = self.net(x)
        if self.with_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ExpandedConv(nn.HybridBlock):
    def __init__(self, inp, oup, t, strides, kernel=3, same_shape=True, **kwargs):
        super(ExpandedConv, self).__init__(**kwargs)

        self.same_shape = same_shape
        self.strides = strides
        with self.name_scope(): 
            self.bottleneck = nn.HybridSequential()
            self.bottleneck.add(
                Conv1x1(inp*t, prefix="expand_"),
                DWise(inp*t, self.strides, kernel, prefix="dwise_"),
                Conv1x1(oup, is_linear=True, prefix="linear_")
            )
    def hybrid_forward(self, F, x):
        out = self.bottleneck(x)
        if self.strides == 1 and self.same_shape:
            out = F.elemwise_add(out, x)
        return out

def ExpandedConvSequence(t, k, inp, oup, repeats, first_strides, **kwargs):
    seq = nn.HybridSequential(**kwargs)
    with seq.name_scope():
        seq.add(ExpandedConv(inp, oup, t, first_strides, k, same_shape=False))
        curr_inp = oup
        for i in range(1, repeats):
            seq.add(ExpandedConv(curr_inp, oup, t, 1))
            curr_inp = oup
    return seq

class MNasNet(nn.HybridBlock):
    def __init__(self, m=1.0, **kwargs):
        super(MNasNet, self).__init__(**kwargs)
        
        self.first_oup = int(32*m)
        self.second_oup = int(16*m)
        #self.second_oup = int(32*m)
        self.interverted_residual_setting = [
            # t, c,  n, s, k
            [3, int(24*m),  3, 2, 3, "stage2_"],  # -> 56x56
            [3, int(40*m),  3, 2, 5, "stage3_"],  # -> 28x28
            [6, int(80*m),  3, 2, 5, "stage4_1_"],  # -> 14x14
            [6, int(96*m),  2, 1, 3, "stage4_2_"],  # -> 14x14
            [6, int(192*m), 4, 2, 5, "stage5_1_"], # -> 7x7
            [6, int(320*m), 1, 1, 3, "stage5_2_"], # -> 7x7          
        ]
        self.last_channels = int(1024*m)

        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(ConvBlock(self.first_oup, 3, 1, prefix="stage1_conv0_"))
            self.features.add(SepCONV(self.first_oup, self.second_oup, 3, prefix="stage1_sepconv0_"))
            inp = self.second_oup
            for i, (t, c, n, s, k, prefix) in enumerate(self.interverted_residual_setting):
                oup = c
                self.features.add(ExpandedConvSequence(t, k, inp, oup, n, s, prefix=prefix))
                inp = oup

            self.features.add(Conv1x1(self.last_channels, prefix="stage5_3_"))
            #self.features.add(nn.GlobalAvgPool2D())
            #self.features.add(nn.Flatten())
            #self.output = nn.Dense(num_classes)
    def hybrid_forward(self, F, x):
        x = self.features(x)
        #x = self.output(x)
        return x

    def num_output_channel(self):
      return self.last_channels

def get_symbol():
  net = MNasNet(config.net_multiplier)
  data = mx.sym.Variable(name='data')
  data = data-127.5
  data = data*0.0078125
  body = net(data)
  fc1 = symbol_utils.get_fc1(body, config.emb_size, config.net_output, input_channel=net.num_output_channel())
  return fc1

