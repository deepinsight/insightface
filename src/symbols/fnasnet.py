
from mxnet.gluon import nn
import mxnet as mx
import symbol_utils

class MaxPoolPad(nn.HybridBlock):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pool = nn.MaxPool2D(pool_size=(3,3), strides=2, padding=1)

    def hybrid_forward(self, F, x):
        x = F.pad(x, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 1, 0, 1, 0))
        x = self.pool(x)
        #x = x[:, :, 1:, 1:]
        x = F.slice(x, begin=(None, None, 1, 1), end=(None, None, None, None))
        return x

class AvgPoolPad(nn.HybridBlock):

    def __init__(self, strides=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pool = nn.AvgPool2D(pool_size=(3,3), strides=strides, padding=padding)

    def hybrid_forward(self, F, x):
        x = F.pad(x, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 1, 0, 1, 0))
        x = self.pool(x)
        #x = x[:, :, 1:, 1:]
        x = F.slice(x, begin=(None, None, 1, 1), end=(None, None, None, None))
        return x

class SeparableConv2d(nn.HybridBlock):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2D(channels=in_channels, kernel_size=dw_kernel,
                                          strides=dw_stride,
                                          padding=dw_padding,
                                          use_bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2D(channels=out_channels, kernel_size=1, strides=1, use_bias=bias)

    def hybrid_forward(self, F, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x

class BranchSeparables(nn.HybridBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.Activation(activation='relu')
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm(epsilon=0.001, momentum=0.1)
        self.relu1 = nn.Activation(activation='relu')
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm(epsilon=0.001, momentum=0.1)

    def hybrid_forward(self, F, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x

class BranchSeparablesStem(nn.HybridBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.Activation(activation='relu')
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm(epsilon=0.001, momentum=0.1)
        self.relu1 = nn.Activation(activation='relu')
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm(epsilon=0.001, momentum=0.1)

    def hybrid_forward(self, F, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x

class BranchSeparablesReduction(BranchSeparables):

    z_padding = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        #self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))
        self.z_padding = z_padding
    def hybrid_forward(self, F, x):
        x = self.relu(x)
        x = F.pad(x, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, self.z_padding, 0, self.z_padding, 0))
        x = self.separable_1(x)
        #x = x[:, :, 1:, 1:]
        x = F.slice(x, begin=(None, None, 1, 1), end=(None, None, None, None))
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x

class CellStem0(nn.HybridBlock):

    def __init__(self):
        super(CellStem0, self).__init__()
        self.conv_1x1 = nn.HybridSequential()
        self.conv_1x1.add(nn.Activation(activation='relu'))
        self.conv_1x1.add(nn.Conv2D(42, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.comb_iter_0_left = BranchSeparables(42, 42, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(96, 42, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(96, 42, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2D(pool_size=3, strides=2, padding=1)
        self.comb_iter_2_right = BranchSeparablesStem(96, 42, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2D(pool_size=3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(42, 42, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

    def hybrid_forward(self, F, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(*[x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out

class CellStem1(nn.HybridBlock):

    def __init__(self):
        super(CellStem1, self).__init__()
        self.conv_1x1 = nn.HybridSequential()
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(channels=84, kernel_size=1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.relu = nn.Activation('relu')
        self.path_1 = nn.HybridSequential()
        self.path_1.add(nn.AvgPool2D(pool_size=1, strides=2))
        self.path_1.add(nn.Conv2D(channels=42, kernel_size=1, strides=1, use_bias=False))

        #self.path_2 = nn.ModuleList()
        #self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        #self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2_avgpool = nn.AvgPool2D(pool_size=1, strides=2)
        self.path_2_conv = nn.Conv2D(channels=42, kernel_size=1, strides=1, use_bias=False)

        self.final_path_bn = nn.BatchNorm(epsilon=0.001, momentum=0.1)

        self.comb_iter_0_left = BranchSeparables(84, 84, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(84, 84, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(84, 84, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2D(pool_size=3, strides=2, padding=1)
        self.comb_iter_2_right = BranchSeparables(84, 84, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2D(pool_size=3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(84, 84, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

    def hybrid_forward(self, F, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = F.pad(x_relu, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 0, 1, 0, 1))
        #x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = F.slice(x_path2, begin=(None, None, 1, 1), end=(None, None, None, None))
        x_path2 = self.path_2_avgpool(x_path2)
        x_path2 = self.path_2_conv(x_path2)
        # final path
        x_right = self.final_path_bn(F.concat(*[x_path1, x_path2], dim=1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(*[x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out

class FirstCell(nn.HybridBlock):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.HybridSequential()
        self.conv_1x1.add(nn.Activation(activation='relu'))
        self.conv_1x1.add(nn.Conv2D(channels=out_channels_right, kernel_size=1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.relu = nn.Activation(activation='relu')
        self.path_1 = nn.HybridSequential()
        self.path_1.add(nn.AvgPool2D(pool_size=1, strides=2))
        self.path_1.add(nn.Conv2D(channels=out_channels_left, kernel_size=1, strides=1, use_bias=False))
        #self.path_2 = nn.ModuleList()
        #self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        #self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2_avgpool = nn.AvgPool2D(pool_size=1, strides=2)
        #self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2_conv = nn.Conv2D(channels=out_channels_left, kernel_size=1, strides=1, use_bias=False)
        self.final_path_bn = nn.BatchNorm(epsilon=0.001, momentum=0.1)

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_3_left = nn.AvgPool2D(3, strides=1, padding=1)
        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def hybrid_forward(self, F, x, x_prev):
        x_relu = self.relu(x_prev)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = F.pad(x_relu, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 0, 1, 0, 1))
        #x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = F.slice(x_path2, begin=(None, None, 1, 1), end=(None, None, None, None))
        x_path2 = self.path_2_avgpool(x_path2)
        x_path2 = self.path_2_conv(x_path2)
        # final path
        x_left = self.final_path_bn(F.concat(*[x_path1, x_path2], dim=1))
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = F.concat(*[x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out

class NormalCell(nn.HybridBlock):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.HybridSequential()
        self.conv_prev_1x1.add(nn.Activation(activation='relu'))
        self.conv_prev_1x1.add(nn.Conv2D(channels=out_channels_left, kernel_size=1, strides=1, use_bias=False))
        self.conv_prev_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.conv_1x1 = nn.HybridSequential()
        self.conv_1x1.add(nn.Activation(activation='relu'))
        self.conv_1x1.add(nn.Conv2D(channels=out_channels_right, kernel_size=1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_3_left = nn.AvgPool2D(3, strides=1, padding=1)
        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = F.concat(*[x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out

class ReductionCell0(nn.HybridBlock):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.HybridSequential()
        self.conv_prev_1x1.add(nn.Activation(activation='relu'))
        self.conv_prev_1x1.add(nn.Conv2D(channels=out_channels_left, kernel_size=1, strides=1, use_bias=False))
        self.conv_prev_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.conv_1x1 = nn.HybridSequential()
        self.conv_1x1.add(nn.Activation(activation='relu'))
        self.conv_1x1.add(nn.Conv2D(channels=out_channels_right, kernel_size=1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(*[x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out

class ReductionCell1(nn.HybridBlock):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.HybridSequential()
        self.conv_prev_1x1.add(nn.Activation(activation='relu'))
        self.conv_prev_1x1.add(nn.Conv2D(channels=out_channels_left, kernel_size=1, strides=1, use_bias=False))
        self.conv_prev_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.conv_1x1 = nn.HybridSequential()
        self.conv_1x1.add(nn.Activation(activation='relu'))
        self.conv_1x1.add(nn.Conv2D(channels=out_channels_right, kernel_size=1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2D(3, strides=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=2, padding=1)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2D(3, strides=2, padding=1)

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(*[x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out

class NASNetALarge(nn.HybridBlock):

    def __init__(self, num_classes=1001):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes

        self.conv0 = nn.HybridSequential()
        self.conv0.add(nn.Conv2D(channels=96, kernel_size=3, padding=0, strides=1, use_bias=False))
        self.conv0.add(nn.BatchNorm(epsilon=0.001, momentum=0.1))

        self.cell_stem_0 = CellStem0()
        self.cell_stem_1 = CellStem1()

        self.cell_0 = FirstCell(in_channels_left=168, out_channels_left=84,
                                in_channels_right=336, out_channels_right=168)
        self.cell_1 = NormalCell(in_channels_left=336, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)
        self.cell_2 = NormalCell(in_channels_left=1008, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)
        self.cell_3 = NormalCell(in_channels_left=1008, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)
        self.cell_4 = NormalCell(in_channels_left=1008, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)
        self.cell_5 = NormalCell(in_channels_left=1008, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)

        self.reduction_cell_0 = ReductionCell0(in_channels_left=1008, out_channels_left=336,
                                               in_channels_right=1008, out_channels_right=336)

        self.cell_6 = FirstCell(in_channels_left=1008, out_channels_left=168,
                                in_channels_right=1344, out_channels_right=336)
        self.cell_7 = NormalCell(in_channels_left=1344, out_channels_left=336,
                                 in_channels_right=2016, out_channels_right=336)
        self.cell_8 = NormalCell(in_channels_left=2016, out_channels_left=336,
                                 in_channels_right=2016, out_channels_right=336)
        self.cell_9 = NormalCell(in_channels_left=2016, out_channels_left=336,
                                 in_channels_right=2016, out_channels_right=336)
        self.cell_10 = NormalCell(in_channels_left=2016, out_channels_left=336,
                                  in_channels_right=2016, out_channels_right=336)
        self.cell_11 = NormalCell(in_channels_left=2016, out_channels_left=336,
                                  in_channels_right=2016, out_channels_right=336)

        self.reduction_cell_1 = ReductionCell1(in_channels_left=2016, out_channels_left=672,
                                               in_channels_right=2016, out_channels_right=672)

        self.cell_12 = FirstCell(in_channels_left=2016, out_channels_left=336,
                                 in_channels_right=2688, out_channels_right=672)
        self.cell_13 = NormalCell(in_channels_left=2688, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)
        self.cell_14 = NormalCell(in_channels_left=4032, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)
        self.cell_15 = NormalCell(in_channels_left=4032, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)
        self.cell_16 = NormalCell(in_channels_left=4032, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)
        self.cell_17 = NormalCell(in_channels_left=4032, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)

        self.relu = nn.Activation(activation='relu')
        self.avgpool = nn.AvgPool2D(pool_size=11, strides=1, padding=0)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.dense= nn.Dense(num_classes)

    def features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)

        return x_cell_17

    def classifier(self, x):
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_symbol(num_classes):
  model = NASNetALarge(num_classes)
  data = mx.sym.Variable(name='data')
  data = data-127.5
  data = data*0.0078125
  body = model.features(data)
  fc1 = symbol_utils.get_fc1(body, num_classes, 'E')
  return fc1

