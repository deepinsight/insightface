# -*- coding: utf-8 -*-
"""

Xception network, suitable for images with around 299 x 299 (original version)

Reference:

François Chollet. Xception: Deep Learning with Depthwise Separable Convlutions. arXiv preprint. https://arxiv.org/pdf/1610.02357v3.pdf

I refered one version of MXNet from u1234x1234 https://github.com/u1234x1234/mxnet-xception/blob/master/symbol_xception.py

Modified by Lin Xiong, Sep-3, 2017 for images 224 x 224
There are some slightly differences with u1234x1234's version (pooling layer) and original version (no dropout layer).

In order to accelerate computation, we use smaller parameters than original paper.

"""

import mxnet as mx
import symbol_utils

def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', withRelu=False, withBn=True, bn_mom=0.9, workspace=256):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                              name='%s%s_conv2d' % (name, suffix), workspace=workspace)
    if withBn:
        conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s%s_bn' % (name, suffix))
    if withRelu:
        conv = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' % (name, suffix))
    return conv

def Separable_Conv(data, num_in_channel, num_out_channel, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None, suffix='', depth_mult=1, withBn=True, bn_mom=0.9, workspace=256):
    # original version of Separable Convolution
    # depthwise convolution
    #channels       = mx.sym.split(data=data, axis=1, num_outputs=num_in_channel) # for new version of mxnet > 0.8
    channels       = mx.sym.SliceChannel(data=data, axis=1, num_outputs=num_in_channel) # for old version of mxnet <= 0.8
    depthwise_outs = [mx.sym.Convolution(data=channels[i], num_filter=depth_mult, kernel=kernel, 
                           stride=stride, pad=pad, name=name+'_depthwise_kernel_'+str(i), workspace=workspace)
                           for i in range(num_in_channel)]
    depthwise_out = mx.sym.Concat(*depthwise_outs)
    # pointwise convolution
    pointwise_out = Conv(data=depthwise_out, num_filter=num_out_channel, name=name+'_pointwise_kernel', withBn=False, bn_mom=0.9, workspace=256)
    if withBn:
        pointwise_out = mx.sym.BatchNorm(data=pointwise_out, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s%s_bn' % (name, suffix))
    return pointwise_out



def Circle_Middle(name, data, 
                  num_filter,
                  bn_mom=0.9,
                  round=8):
    b = data
    for i in xrange(round):
        residual = b
        prefix = name + '_block' + ('_%d' % i)

        b = mx.sym.Activation(data=b, act_type='relu', name=prefix + '_sepconv1_relu')
        b = Separable_Conv(data=b, num_in_channel=num_filter, num_out_channel=num_filter, name=prefix + '_sepconv1', withBn=True, bn_mom=bn_mom, workspace=256)
        b = mx.sym.Activation(data=b, act_type='relu', name=prefix + '_sepconv2_relu')
        b = Separable_Conv(data=b, num_in_channel=num_filter, num_out_channel=num_filter, name=prefix + '_sepconv2', withBn=True, bn_mom=bn_mom, workspace=256)
        b = mx.sym.Activation(data=b, act_type='relu', name=prefix + '_sepconv3_relu')
        b = Separable_Conv(data=b, num_in_channel=num_filter, num_out_channel=num_filter, name=prefix + '_sepconv3', withBn=True, bn_mom=bn_mom, workspace=256)

        b = b + residual

    return b


def get_symbol(num_classes=1000, **kwargs):
    # input shape 229*229*3 (old)
    # input shape 224*224*3 (new)
    
    #filter_list=[64, 128, 256, 728, 1024, 1536, 2048]     # original version
    filter_list=[64, 64, 128, 364, 512, 768, 1024]  # smaller one

    # Entry flow
    data = mx.sym.Variable('data')
    data = data-127.5
    data = data*0.0078125
    version_input = kwargs.get('version_input',1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    print(version_input, version_output, version_unit)

    if version_input>=2:
      filter_list=[64, 128, 256, 728, 1024, 1536, 2048]     # original version

    # block 1
    if version_input==0:
      block1 = Conv(data=data, num_filter=int(filter_list[0]*0.5), kernel=(3, 3), stride=(2, 2), pad=(1, 1), name='Entry_flow_b1_conv1', 
                    withRelu=True, withBn=True, bn_mom=0.9, workspace=256)
    else:
      block1 = Conv(data=data, num_filter=int(filter_list[0]*0.5), kernel=(3, 3), stride=(1,1), pad=(1, 1), name='Entry_flow_b1_conv1', 
                    withRelu=True, withBn=True, bn_mom=0.9, workspace=256)
    block1 = Conv(data=block1, num_filter=filter_list[0], kernel=(3, 3), pad=(1, 1), name='Entry_flow_b1_conv2',
                  withRelu=True, withBn=True, bn_mom=0.9, workspace=256)

    # block 2
    rs2    = Conv(data=block1, num_filter=filter_list[1], stride=(2, 2), name='Entry_flow_b2_conv1',
                  withBn=True, bn_mom=0.9, workspace=256)
    block2 = Separable_Conv(block1, num_in_channel=filter_list[0], num_out_channel=filter_list[1], name='Entry_flow_b2_sepconv1', withBn=True, bn_mom=0.9, workspace=256)
    block2 = mx.sym.Activation(data=block2, act_type='relu', name='Entry_flow_b2_sepconv1_relu')
    block2 = Separable_Conv(block2, num_in_channel=filter_list[1], num_out_channel=filter_list[1], name='Entry_flow_b2_sepconv2', withBn=True, bn_mom=0.9, workspace=256)
    block2 = mx.sym.Pooling(data=block2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='Entry_flow_b2_pool')
    block2 = block2 + rs2

    # block 3
    rs3    = Conv(data=block2, num_filter=filter_list[2], stride=(2, 2), name='Entry_flow_b3_conv1',
                  withBn=True, bn_mom=0.9, workspace=256)
    block3 = mx.sym.Activation(data=block2, act_type='relu', name='Entry_flow_b3_sepconv1_relu')
    block3 = Separable_Conv(block3, num_in_channel=filter_list[1], num_out_channel=filter_list[2], name='Entry_flow_b3_sepconv1', withBn=True, bn_mom=0.9, workspace=256)
    block3 = mx.sym.Activation(data=block3, act_type='relu', name='Entry_flow_b3_sepconv2_relu')
    block3 = Separable_Conv(block3, num_in_channel=filter_list[2], num_out_channel=filter_list[2], name='Entry_flow_b3_sepconv2', withBn=True, bn_mom=0.9, workspace=256)
    block3 = mx.sym.Pooling(data=block3, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='Entry_flow_b3_pool')
    block3 = block3 + rs3

    # block 4
    rs4    = Conv(data=block3, num_filter=filter_list[3], stride=(2, 2), name='Entry_flow_b4_conv1',
                  withBn=True, bn_mom=0.9, workspace=256)
    block4 = mx.sym.Activation(data=block3, act_type='relu', name='Entry_flow_b4_sepconv1_relu')
    block4 = Separable_Conv(block4, num_in_channel=filter_list[2], num_out_channel=filter_list[3], name='Entry_flow_b4_sepconv1', withBn=True, bn_mom=0.9, workspace=256)
    block4 = mx.sym.Activation(data=block4, act_type='relu', name='Entry_flow_b4_sepconv2_relu')
    block4 = Separable_Conv(block4, num_in_channel=filter_list[3], num_out_channel=filter_list[3], name='Entry_flow_b4_sepconv2', withBn=True, bn_mom=0.9, workspace=256)
    block4 = mx.sym.Pooling(data=block4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='Entry_flow_b4_pool')
    block4 = block4 + rs4

    # Middle flow
    block_m_f = Circle_Middle('Middle_flow', block4,
                      filter_list[3],
                      0.9,
                      8)
    # Exit flow
    rs5    = Conv(data=block_m_f, num_filter=filter_list[4], stride=(2, 2), name='Exit_flow_b5_conv1',
                  withBn=True, bn_mom=0.9, workspace=256)
    block5 = mx.sym.Activation(data=block_m_f, act_type='relu', name='Exit_flow_b5_sepconv1_relu')
    block5 = Separable_Conv(block5, num_in_channel=filter_list[3], num_out_channel=filter_list[3], name='Exit_flow_b5_sepconv1', withBn=True, bn_mom=0.9, workspace=256)
    block5 = mx.sym.Activation(data=block5, act_type='relu', name='Exit_flow_b5_sepconv2_relu')
    block5 = Separable_Conv(block5, num_in_channel=filter_list[3], num_out_channel=filter_list[4], name='Exit_flow_b5_sepconv2', withBn=True, bn_mom=0.9, workspace=256)
    block5 = mx.sym.Pooling(data=block5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='Entry_flow_b5_pool')
    block5 = block5 + rs5

    block6 = Separable_Conv(block5, num_in_channel=filter_list[4], num_out_channel=filter_list[5], name='Exit_flow_b6_sepconv1', withBn=True, bn_mom=0.9, workspace=256)
    block6 = mx.sym.Activation(data=block6, act_type='relu', name='Exit_flow_b6_sepconv1_relu')
    block6 = Separable_Conv(block6, num_in_channel=filter_list[5], num_out_channel=filter_list[6], name='Exit_flow_b6_sepconv2', withBn=True, bn_mom=0.9, workspace=256)
    block6 = mx.sym.Activation(data=block6, act_type='relu', name='Exit_flow_b6_sepconv2_relu')
    fc1 = symbol_utils.get_fc1(block6, num_classes, fc_type)
    return fc1


