import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
import symbol_utils

def ConvBlock(channels, kernel_size, strides):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, kernel_size, strides=strides, padding=1, use_bias=False),
        nn.BatchNorm(scale=True),
        nn.Activation('relu')
    )
    return out

def Conv1x1(channels, is_linear=False):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, 1, padding=0, use_bias=False),
        nn.BatchNorm(scale=True)
    )
    if not is_linear:
        out.add(nn.Activation('relu'))
    return out

def DWise(channels, stride):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, 3, strides=stride, padding=1, groups=channels, use_bias=False),
        nn.BatchNorm(scale=True),
        nn.Activation('relu')
    )
    return out

class InvertedResidual(nn.HybridBlock):
    def __init__(self, t, e, c, s, same_shape=True, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.stride = s
        with self.name_scope(): 
            self.bottleneck = nn.HybridSequential()
            self.bottleneck.add(
                Conv1x1(e*t),
                DWise(e*t, self.stride),
                Conv1x1(c, is_linear=True)
            )
            if self.stride == 1 and not self.same_shape:
                self.conv_res = Conv1x1(c)
    def hybrid_forward(self, F, x):
        out = self.bottleneck(x)
        #if self.stride == 1 and self.same_shape:
        #    out = F.elemwise_add(out, x)
        if self.stride == 1:
            if not self.same_shape:
                x = self.conv_res(x)
            out = F.elemwise_add(out, x)
        return out

class MobilenetV2(nn.HybridBlock):
    def __init__(self, num_classes=1000, width_mult=1.0, **kwargs):
        super(MobilenetV2, self).__init__(**kwargs)
        
        self.w = width_mult
        
        self.cn = [int(x*self.w) for x in [32, 16, 24, 32, 64, 96, 160, 320]]
        
        def InvertedResidualSequence(t, cn_id, n, s):
            seq = nn.HybridSequential()
            seq.add(InvertedResidual(t, self.cn[cn_id-1], self.cn[cn_id], s, same_shape=False))
            for _ in range(n-1):
                seq.add(InvertedResidual(t, self.cn[cn_id-1], self.cn[cn_id], 1))
            return seq
        
        self.b0 = ConvBlock(self.cn[0], 3, 1)
        self.b1 = InvertedResidualSequence(1, 1, 1, 1)
        self.b2 = InvertedResidualSequence(6, 2, 2, 2)
        self.b3 = InvertedResidualSequence(6, 3, 3, 2)
        self.b4 = InvertedResidualSequence(6, 4, 4, 1)
        self.b5 = InvertedResidualSequence(6, 5, 3, 2)
        self.b6 = InvertedResidualSequence(6, 6, 3, 2)
        self.b7 = InvertedResidualSequence(6, 7, 1, 1)

        self.last_channels = int(1280*self.w) if self.w > 1.0 else 1280
        with self.name_scope():
            self.features = nn.HybridSequential()
            with self.features.name_scope():
                self.features.add(self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)
                self.features.add(Conv1x1(self.last_channels))
                #self.features.add(nn.GlobalAvgPool2D())
                #self.features.add(nn.Flatten())
            #self.output = nn.Dense(num_classes)
    def hybrid_forward(self, F, x):
        x = self.features(x)
        #x = self.output(x)
        return x

def get_symbol(num_classes):
  net = MobilenetV2(num_classes, 1)
  data = mx.sym.Variable(name='data')
  data = data-127.5
  data = data*0.0078125
  body = net(data)
  fc1 = symbol_utils.get_fc1(body, num_classes, 'E')
  return fc1

