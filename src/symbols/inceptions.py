# -*- coding:utf-8 -*-
__author__ = 'zhangshuai'
modified_date = '16/7/5'
__modify__ = 'anchengwu'
modified_date = '17/2/22'
__modify2__ = 'weiyangwang'
modified_date = '17/9/20'


'''
Inception v4 , suittable for image with around 299 x 299

Reference:
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke
    arXiv.1602.07261
    
    
Inception V3, suitable for images with around 299 x 299

Reference:

Szegedy, Christian, et al. "Rethinking the Inception Architecture for Computer Vision." arXiv preprint arXiv:1512.00567 (2015).
    
'''


# --------------------------------------------------------

# Modified By DeepInsight

#  0. Make Code Tidier (with exec)
#  1. Scalable Inception V3, V4, -resnetV2
#  2. Todo: Modified For XCeption, make Conv11 num_group_11 and Other Conv num_group independent.
#  3. Todo: Module Options: Deformable, Attention Along Features/Along Image
#  4. Todo: Adaptive Encoder-Decoder Symbol For Segmenter
#  5. Todo: Adaptive Symbol For Detector

# --------------------------------------------------------


import mxnet as mx
import numpy as np

######## Inception Common:

## Todo: Deformable, Attention

def Conv(data, num_filter, num_group = 1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), \
         act_type="relu", mirror_attr={}, with_act=True, name=None, suffix=''):
    
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    if with_act:
        act = mx.sym.Activation(data=bn, act_type=act_type, name='%s%s_relu' %(name, suffix))
        return act
    else:
        return bn
    
def get_input_size(lastout=8):
    input_size = 2*lastout + 1 # 17
    input_size = 2*input_size + 1 # 35
    input_size = 2*input_size + 1 # 71
    input_size = input_size + 2 # 73
    input_size = 2*input_size + 1 # 147
    input_size = input_size + 2 # 149
    input_size = 2*input_size + 1 # 299
    return input_size
    

######## Inception ResNetv2: Scalable, XCeptionized

# Todo Scalable and XCeptionized

''' Fade-away ConvFactory

def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True):
    conv = mx.symbol.Convolution(
        data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    if with_act:
        act = mx.symbol.Activation(
            data=bn, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return bn
'''

def block35_irv2(net, input_num_channels, 
                 basefilter=16, num_group=1 ,num_group_11=1, scale=1.0,
                 with_act=True, act_type='relu', mirror_attr={}, name=None):
    #Conv11
    tower_conv = Conv(net, basefilter*2, num_group=num_group_11, kernel=(1, 1), name=name+'_35b11')
    #Conv11-Conv33
    tower_conv1_0 = Conv(net, basefilter*2, num_group=num_group_11, kernel=(1, 1), name=name+'_35b21')
    tower_conv1_1 = Conv(tower_conv1_0, basefilter*2, num_group=num_group, kernel=(3, 3), pad=(1, 1), name=name+'_35b22')
    #Conv11-Conv33-Conv33
    tower_conv2_0 = Conv(net, basefilter*2, num_group=num_group_11,kernel=(1, 1), name=name+'_35b31')
    tower_conv2_1 = Conv(tower_conv2_0, basefilter*3, num_group=num_group, kernel=(3, 3), pad=(1, 1), name=name+'_35b32')
    tower_conv2_2 = Conv(tower_conv2_1, basefilter*4, num_group=num_group, kernel=(3, 3), pad=(1, 1), name=name+'_35b33')
    #Concat
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
    tower_out = Conv(tower_mixed, input_num_channels, num_group=num_group_11, kernel=(1, 1), with_act=False, name=name+'_35out')
    
    
    net = net + tower_out * scale
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block17_irv2(net, input_num_channels,
                 basefilter=32, num_group=1 ,num_group_11=1, scale=1.0,
                 with_act=True, act_type='relu', mirror_attr={}, name=None):
    # Conv11
    tower_conv = Conv(net, basefilter*6, num_group=num_group_11, kernel=(1, 1), name=name+'_17b11')
    # Conv11-Conv17-Conv71
    tower_conv1_0 = Conv(net, basefilter*6, num_group=num_group_11, kernel=(1, 1), name=name+'_17b21')
    tower_conv1_1 = Conv(tower_conv1_0, basefilter*5, num_group=num_group, kernel=(1, 7), pad=(1, 2), name=name+'_17b22')
    tower_conv1_2 = Conv(tower_conv1_1, basefilter*6, num_group=num_group, kernel=(7, 1), pad=(2, 1), name=name+'_17b23')
    # Concat
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    # Conv11
    tower_out = Conv(
        tower_mixed, input_num_channels, num_group=num_group_11, kernel=(1, 1), with_act=False, name=name+'_17out')
    net = net + tower_out * scale
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block8_irv2(net, input_num_channels,
                basefilter=32, num_group=1 ,num_group_11=1, scale=1.0,
                with_act=True, act_type='relu', mirror_attr={}, name=None):
    # Conv11
    tower_conv = Conv(net, basefilter*6, num_group=num_group_11, kernel=(1, 1), name=name+'_8b11')
    # Conv11-Conv13-Conv31
    tower_conv1_0 = Conv(net, basefilter*6, num_group=num_group_11, kernel=(1, 1), name=name+'_8b21')
    tower_conv1_1 = Conv(tower_conv1_0, basefilter*7, num_group=num_group, kernel=(1, 3), pad=(0, 1), name=name+'_8b22')
    tower_conv1_2 = Conv(tower_conv1_1, basefilter*8, num_group=num_group, kernel=(3, 1), pad=(1, 0), name=name+'_8b23')
    #Concat
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    #Conv11
    tower_out = Conv(
        tower_mixed, input_num_channels, num_group=num_group_11, kernel=(1, 1), with_act=False, name=name+'_8out')
    
    net = net + tower_out * scale
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def repeat(inputs, repetitions, layer, name=None, *args, **kwargs):
    outputs = inputs
    for i in range(repetitions):
        outputs = layer(outputs, name=name+'_'+str(i), *args, **kwargs)
    return outputs


def get_symbol_irv2(num_classes=1000, 
               basefilter=16, num_group=1 ,num_group_11=1, scale=1.0,
               lastout = 8,
               units = [10,20,9],
               **kwargs):
    data = mx.symbol.Variable(name='data')
    # Size 299
    # Stem 1 And Downsampling
    conv1a_3_3 = Conv(data,
                      basefilter*2, num_group=num_group,
                      kernel=(3, 3), stride=(2, 2), name='conv1a')
    # Size 149
    conv2a_3_3 = Conv(conv1a_3_3, basefilter*2, num_group=num_group, kernel=(3, 3), name='conv2a')
    # Size 147
    conv2b_3_3 = Conv(conv2a_3_3, basefilter*4, num_group=num_group, kernel=(3, 3), pad=(1, 1), name='conv2b')
    # Size 147
    maxpool3a_3_3 = mx.symbol.Pooling(
        data=conv2b_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')
    # Stem 2 And Downsampling
    conv3b_1_1 = Conv(maxpool3a_3_3, basefilter*5, num_group=num_group_11, kernel=(1, 1), name='conv3b')
    # 73
    conv4a_3_3 = Conv(conv3b_1_1, basefilter*12, num_group=num_group, kernel=(3, 3), name='conv4a')
    # 71
    maxpool5a_3_3 = mx.symbol.Pooling(
        data=conv4a_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')
  
    # Size 35
    # Stem 3 And Downsampling
    # Branch31: Conv11
    tower_conv = Conv(maxpool5a_3_3, basefilter*6, num_group=num_group_11, kernel=(1, 1), name='branch31')
    # Branch32: Conv11-Conv55
    tower_conv1_0 = Conv(maxpool5a_3_3, basefilter*3, num_group=num_group_11, kernel=(1, 1), name='branch321')
    tower_conv1_1 = Conv(tower_conv1_0, basefilter*4, num_group=num_group, kernel=(5, 5), pad=(2, 2), name='branch322')
    # Branch33: Conv11-Conv33-Conv33
    tower_conv2_0 = Conv(maxpool5a_3_3, basefilter*4, num_group=num_group_11, kernel=(1, 1), name='branch331')
    tower_conv2_1 = Conv(tower_conv2_0, basefilter*6, num_group=num_group, kernel=(3, 3), pad=(1, 1), name='branch332')
    tower_conv2_2 = Conv(tower_conv2_1, basefilter*6, num_group=num_group, kernel=(3, 3), pad=(1, 1), name='branch333')
    # Branch34: Pool-Conv11
    tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    tower_conv3_1 = Conv(tower_pool3_0, basefilter*4, num_group=num_group_11, kernel=(1, 1),name='branch34')
    # Concat
    tower_5b_out = mx.symbol.Concat(
        *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
    
    # Repeat 1
    net = repeat(tower_5b_out, units[0], block35_irv2, scale=0.17, input_num_channels=basefilter*20,\
                basefilter=basefilter, num_group=num_group ,num_group_11=num_group_11, name='repeat1')
    
    # Size 35
    
    # Branch 41
    tower_conv = Conv(net, basefilter*24, num_group=num_group, kernel=(3, 3), stride=(2, 2), name='branch41')
    # Branch 42
    tower_conv1_0 = Conv(net, basefilter*16, num_group=num_group_11, kernel=(1, 1), name='branch421')
    tower_conv1_1 = Conv(tower_conv1_0, basefilter*16, num_group=num_group, kernel=(3, 3), pad=(1, 1), name='branch422')
    tower_conv1_2 = Conv(tower_conv1_1, basefilter*24, num_group=num_group, kernel=(3, 3), stride=(2, 2), name='branch423')
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    
    # Concat 
    net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])
    # Repeat 2
    net = repeat(net, units[1], block17_irv2, scale=0.1, input_num_channels=basefilter*68,\
                basefilter=basefilter*2, num_group=num_group, num_group_11=num_group_11,name='repeat2')
    
    
    # Size 17
    
    # Branch51: Conv11-Conv33
    tower_conv = Conv(net, basefilter*16, num_group=num_group_11, kernel=(1, 1) ,name='branch511')
    tower_conv0_1 = Conv(tower_conv, basefilter*24, num_group=num_group, kernel=(3, 3), stride=(2, 2) ,name='branch512')
    # Branch52: Conv11-Conv33 ? Is this XCeption
    tower_conv1 = Conv(net, basefilter*16, num_group=num_group_11, kernel=(1, 1) ,name='branch521')
    tower_conv1_1 = Conv(tower_conv1, basefilter*18, num_group=num_group, kernel=(3, 3), stride=(2, 2) ,name='branch522')
    # Branch53: Conv11-Conv33-Conv33
    tower_conv2 = Conv(net, basefilter*16, num_group=num_group_11, kernel=(1, 1) ,name='branch531')
    tower_conv2_1 = Conv(tower_conv2, basefilter*18,  num_group=num_group, kernel=(3, 3), pad=(1, 1) ,name='branch532')
    tower_conv2_2 = Conv(tower_conv2_1, basefilter*20,  num_group=num_group, kernel=(3, 3),  stride=(2, 2) ,name='branch533')
    # Pool33
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(
        *[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])

    # Size 8 
    net = repeat(net, units[2], block8_irv2, scale=0.2, input_num_channels=basefilter*130,\
                basefilter=basefilter*2, num_group=num_group ,num_group_11=num_group_11,name='repeat3')
    net = block8_irv2(net, with_act=False, input_num_channels=basefilter*130,
                     basefilter=basefilter*2, num_group=num_group ,num_group_11=num_group_11,name='block8')
    
    # Trailing
    net = Conv(net, basefilter*96, num_group=num_group_11, kernel=(1, 1), name='trailing')
    net = mx.symbol.Pooling(net, kernel=(
        1, 1), global_pool=True, stride=(2, 2), pool_type='avg')
    net = mx.symbol.Flatten(net)
    net = mx.symbol.Dropout(data=net, p=0.2)
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    return net, softmax



######## Inception V4: Scalable, XCeptionized

def Inception_stem_V4(data, basefilter=32, stem_num_group=1, stem_num_group_11=1, name= None):
    
    # Size 299
    c = Conv(data, basefilter, num_group=stem_num_group, kernel=(3, 3), stride=(2, 2), name='%s_conv1_3*3' %name)
    # 149
    c = Conv(c, basefilter, num_group=stem_num_group, kernel=(3, 3), name='%s_conv2_3*3' %name)
    # 147
    c = Conv(c, basefilter, num_group=stem_num_group, kernel=(3, 3), pad=(1, 1), name='%s_conv3_3*3' %name)
    # 147
    p1 = mx.sym.Pooling(c, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)
    # 73 
    c2 = Conv(c, basefilter*3, num_group=stem_num_group, kernel=(3, 3), stride=(2, 2), name='%s_conv4_3*3' %name)
    concat = mx.sym.Concat(*[p1, c2], name='%s_concat_1' %name)

    c1 = Conv(concat, basefilter*2, num_group=stem_num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv5_1*1' %name)
    c1 = Conv(c1, basefilter*3, num_group=stem_num_group, kernel=(3, 3), name='%s_conv6_3*3' %name)
    
    # 71
    
    c2 = Conv(concat, basefilter*2, num_group=stem_num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv7_1*1' %name)
    c2 = Conv(c2, basefilter*2, num_group=stem_num_group, kernel=(7, 1), pad=(3, 0), name='%s_conv8_7*1' %name)
    c2 = Conv(c2, basefilter*2, num_group=stem_num_group, kernel=(1, 7), pad=(0, 3), name='%s_conv9_1*7' %name)
    c2 = Conv(c2, basefilter*3, num_group=stem_num_group, kernel=(3, 3), pad=(0, 0), name='%s_conv10_3*3' %name)

    concat = mx.sym.Concat(*[c1, c2], name='%s_concat_2' %name)

    c1 = Conv(concat, basefilter*6, num_group=stem_num_group, kernel=(3, 3), stride=(2, 2), name='%s_conv11_3*3' %name)
    p1 = mx.sym.Pooling(concat, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_2' %name)

    # 35
    
    concat = mx.sym.Concat(*[c1, p1], name='%s_concat_3' %name)
    return concat


def InceptionA_V4(input, basefilter=32, num_group=1 ,num_group_11=1,  name=None):
    # Pool33-Conv11
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = Conv(p1, basefilter*3, kernel=(1, 1), num_group=num_group_11, pad=(0, 0), name='%s_conv1_1*1' %name)
    # Conv11
    c2 = Conv(input, basefilter*3, kernel=(1, 1), num_group=num_group_11, pad=(0, 0), name='%s_conv2_1*1' %name)
    # Conv11-Conv33
    c3 = Conv(input, basefilter*2, kernel=(1, 1), num_group=num_group_11, pad=(0, 0), name='%s_conv3_1*1' %name)
    c3 = Conv(c3, basefilter*3, kernel=(3, 3), num_group=num_group, pad=(1, 1), name='%s_conv4_3*3' %name)
    # Conv11-Conv33-Conv33
    c4 = Conv(input, basefilter*2, kernel=(1, 1), num_group=num_group_11, pad=(0, 0), name='%s_conv5_1*1' % name)
    c4 = Conv(c4, basefilter*3, kernel=(3, 3), num_group=num_group, pad=(1, 1), name='%s_conv6_3*3' % name)
    c4 = Conv(c4, basefilter*3, kernel=(3, 3), num_group=num_group, pad=(1, 1), name='%s_conv7_3*3' %name)
    
    concat = mx.sym.Concat(*[c1, c2, c3, c4], name='%s_concat_1' %name)
    return concat


def ReductionA_V4(input, basefilter=32, num_group=1, num_group_11=1, name=None):
    # Pool33
    p1 = mx.sym.Pooling(input, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)
    # Conv33
    c2 = Conv(input, basefilter*12, num_group=num_group, kernel=(3, 3), stride=(2, 2), name='%s_conv1_3*3' %name)
    # Conv11-Conv33-Conv33
    c3 = Conv(input, basefilter*6, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)
    c3 = Conv(c3, basefilter*7, num_group=num_group, kernel=(3, 3), pad=(1, 1), name='%s_conv3_3*3' %name)
    c3 = Conv(c3, basefilter*8, num_group=num_group, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name='%s_conv4_3*3' %name)

    concat = mx.sym.Concat(*[p1, c2, c3], name='%s_concat_1' %name)

    return concat

def InceptionB_V4(input, basefilter=32, num_group=1, num_group_11=1, name=None):
    # Pool33-Conv11
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = Conv(p1, basefilter*4, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)
    # Conv11
    c2 = Conv(input, basefilter*12, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)
    # Conv11-Conv17-Conv71
    c3 = Conv(input, basefilter*6, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv3_1*1' %name)
    c3 = Conv(c3, basefilter*7, num_group=num_group, kernel=(1, 7), pad=(0, 3), name='%s_conv4_1*7' %name)
    #paper wrong
    c3 = Conv(c3, basefilter*8, num_group=num_group, kernel=(7, 1), pad=(3, 0), name='%s_conv5_1*7' %name)
    
    # COnv11-Conv17-Conv71-Conv17-Conv71
    c4 = Conv(input, basefilter*6, kernel=(1, 1), pad=(0, 0), name='%s_conv6_1*1' %name)
    c4 = Conv(c4, basefilter*6, num_group=num_group, kernel=(1, 7), pad=(0, 3), name='%s_conv7_1*7' %name)
    c4 = Conv(c4, basefilter*7, num_group=num_group, kernel=(7, 1), pad=(3, 0), name='%s_conv8_7*1' %name)
    c4 = Conv(c4, basefilter*7, num_group=num_group, kernel=(1, 7), pad=(0, 3), name='%s_conv9_1*7' %name)
    c4 = Conv(c4, basefilter*8, num_group=num_group, kernel=(7, 1), pad=(3, 0), name='%s_conv10_7*1' %name)

    concat = mx.sym.Concat(*[c1, c2, c3, c4], name='%s_concat_1' %name)

    return concat

def ReductionB_V4(input, basefilter=64, num_group=1, num_group_11=1,  name=None):
    # Pool33
    p1 = mx.sym.Pooling(input, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)
    # Conv11-Conv33
    c2 = Conv(input, basefilter*3 , num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)
    c2 = Conv(c2, basefilter*3, num_group=num_group, kernel=(3, 3), stride=(2, 2), name='%s_conv2_3*3' %name)
    # Conv11-Conv17-Conv71-Conv33
    c3 = Conv(input, basefilter*3, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv3_1*1' %name)
    c3 = Conv(c3, basefilter*4, num_group=num_group, kernel=(1, 7), pad=(0, 3), name='%s_conv4_1*7' %name)
    c3 = Conv(c3, basefilter*5, num_group=num_group, kernel=(7, 1), pad=(3, 0), name='%s_conv5_7*1' %name)
    c3 = Conv(c3, basefilter*5, num_group=num_group, kernel=(3, 3), stride=(2, 2), name='%s_conv6_3*3' %name)

    concat = mx.sym.Concat(*[p1, c2, c3], name='%s_concat_1' %name)

    return concat


def InceptionC_V4(input, basefilter=64, num_group=1, num_group_11=1, name=None):
    # Pool33-Conv11
    p1 = mx.sym.Pooling(input, kernel=(3, 3), pad=(1, 1), pool_type='avg', name='%s_avgpool_1' %name)
    c1 = Conv(p1, basefilter*4, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)
    # Conv11
    c2 = Conv(input, basefilter*4, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)
    # Conv11-[Conv13;Conv31]
    c3 = Conv(input, basefilter*6, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv3_1*1' %name)
    c3_1 = Conv(c3, basefilter*4, num_group=num_group, kernel=(1, 3), pad=(0, 1), name='%s_conv4_3*1' %name)
    c3_2 = Conv(c3, basefilter*4, num_group=num_group, kernel=(3, 1), pad=(1, 0), name='%s_conv5_1*3' %name)
    # Conv11-Conv13-Conv31-[Conv13;Conv31]
    c4 = Conv(input, basefilter*6, num_group=num_group_11, kernel=(1, 1), pad=(0, 0), name='%s_conv6_1*1' %name)
    c4 = Conv(c4, basefilter*7, num_group=num_group, kernel=(1, 3), pad=(0, 1), name='%s_conv7_1*3' %name)
    c4 = Conv(c4, basefilter*8, num_group=num_group, kernel=(3, 1), pad=(1, 0), name='%s_conv8_3*1' %name)
    c4_1 = Conv(c4, basefilter*4, num_group=num_group, kernel=(3, 1), pad=(1, 0), name='%s_conv9_1*3' %name)
    c4_2 = Conv(c4, basefilter*4, num_group=num_group, kernel=(1, 3), pad=(0, 1), name='%s_conv10_3*1' %name)

    concat = mx.sym.Concat(*[c1, c2, c3_1, c3_2, c4_1, c4_2], name='%s_concat' %name)

    return concat


def get_symbol_V4(num_classes=1000, \
                  units=[4,7,3], basefilter=32, num_group=1, num_group_11=1, \
                  lastout=8,
                  dtype='float32', **kwargs):
    data = mx.sym.Variable(name="data")
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    x = Inception_stem_V4(data, 
                          basefilter=basefilter,
                          stem_num_group=num_group,
                          stem_num_group_11=num_group_11,
                          name='in_stem')

    #4 * InceptionA By Default

    for i in range(units[0]):
        x = InceptionA_V4(x,
                          basefilter=basefilter,
                          num_group=num_group,
                          num_group_11=num_group_11,
                          name='in%dA' %(i+1))

    #Reduction A : Size 35-17
    x = ReductionA_V4(x,
                      basefilter=basefilter,
                      num_group=num_group,
                      num_group_11=num_group_11,
                      name='re1A')

    #7 * InceptionB By Default

    for i in range(units[1]):
        x = InceptionB_V4(x,
                          basefilter=basefilter,
                          num_group=num_group,
                          num_group_11=num_group_11,
                          name='in%dB' %(i+1))

    #ReductionB : Size 17-8
    x = ReductionB_V4(x,
                      basefilter=basefilter*2,
                      num_group=num_group,
                      num_group_11=num_group_11,
                      name='re1B')

    #3 * InceptionC By Default

    for i in range(units[2]):
        x = InceptionC_V4(x,
                          basefilter=basefilter*2,
                          num_group=num_group,
                          num_group_11=num_group_11,
                          name='in%dC' %(i+1))

    #Average Pooling
    x = mx.sym.Pooling(x, kernel=(lastout, lastout), pad=(1, 1), pool_type='avg', name='global_avgpool')

    #Dropout
    x = mx.sym.Dropout(x, p=0.2)

    flatten = mx.sym.Flatten(x, name='flatten')
    fc1 = mx.sym.FullyConnected(flatten, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    softmax = mx.sym.SoftmaxOutput(fc1, name='softmax')

    return softmax



######## Inception V3: Scalable, XCeptionized

# First Stage
def Inception7A_V3(data,
                   basefilter=16,  #
                   num_filters=[], # Length-7
                   num_group=1, num_group_11=1,
                   pool='avg', name=''):
    assert len(num_filters)==7
    num_1x1, num_3x3_red, num_3x3_1, num_3x3_2, num_5x5_red, num_5x5, proj = tuple( num_filters )
    # Branch 1 : Conv11
    tower_1x1 = Conv(data, basefilter*num_1x1, num_group=num_group_11,  name=('%s_conv' % name))
    # Branch 2 : Conv11-Conv55
    tower_5x5 = Conv(data, basefilter*num_5x5_red, num_group=num_group_11,  name=('%s_tower' % name), suffix='_conv')
    tower_5x5 = Conv(tower_5x5, basefilter*num_5x5, num_group=num_group, kernel=(5, 5), pad=(2, 2), name=('%s_tower' % name), suffix='_conv_1')
    # Branch 3 : Conv11-Conv33-Conv33
    tower_3x3 = Conv(data, basefilter*num_3x3_red, num_group=num_group_11, name=('%s_tower_1' % name), suffix='_conv')
    tower_3x3 = Conv(tower_3x3, basefilter*num_3x3_1, num_group=num_group, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_3x3 = Conv(tower_3x3, basefilter*num_3x3_2, num_group=num_group, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_2')
    # Branch 4: Pool33-Conv11
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(pooling, basefilter*proj, num_group=num_group_11, name=('%s_tower_2' %  name), suffix='_conv')
    concat = mx.sym.Concat(*[tower_1x1, tower_5x5, tower_3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat


# First Downsample

# Field: (x-2)/2, original 38

def Inception7B_V3(data,
                basefilter=32, # Base=32
                num_filters=[], # Length-4
                num_group=1, num_group_11=1,
                pool="max",
                name=''):
    
    assert len(num_filters)==4          
    num_3x3, num_d3x3_red, num_d3x3_1, num_d3x3_2 = tuple(num_filters)
    
    # Branch 1: Conv33
    tower_3x3 = Conv(data, basefilter*num_3x3, num_group=num_group, kernel=(3, 3), pad=(0, 0), stride=(2, 2), name=('%s_conv' % name))
    # Branch 2: Conv11-Conv33-Conv33
    tower_d3x3 = Conv(data, basefilter*num_d3x3_red, num_group=num_group_11, name=('%s_tower' % name), suffix='_conv')
    tower_d3x3 = Conv(tower_d3x3, basefilter*num_d3x3_1, num_group=num_group, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_tower' % name), suffix='_conv_1')
    tower_d3x3 = Conv(tower_d3x3, basefilter*num_d3x3_2, num_group=num_group, kernel=(3, 3), pad=(0, 0), stride=(2, 2), name=('%s_tower' % name), suffix='_conv_2')
    # Branch 3: Pool33
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0,0), pool_type="max", name=('max_pool_%s_pool' % name))
    concat = mx.sym.Concat(*[tower_3x3, tower_d3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat


# Second Stage

def Inception7C_V3(data,
                basefilter=32, 
                num_filters=[], # Length-10
                num_group=1, num_group_11=1,
                pool = 'avg',
                name = ''):
    
    assert len(num_filters)==10
    num_1x1, num_d7_red, num_d7_1, num_d7_2, num_q7_red, \
        num_q7_1, num_q7_2, num_q7_3, num_q7_4, proj = tuple(num_filters)
    
    # Branch 1 : Conv11
    tower_1x1 = Conv(data=data, num_filter=basefilter*num_1x1, kernel=(1, 1), name=('%s_conv' % name))
    # Branch 2: Conv11-Conv17-Conv71
    tower_d7 = Conv(data=data, num_filter=basefilter*num_d7_red, name=('%s_tower' % name), suffix='_conv')
    tower_d7 = Conv(data=tower_d7, num_filter=basefilter*num_d7_1, num_group=num_group, kernel=(1, 7), pad=(0, 3), name=('%s_tower' % name), suffix='_conv_1')
    tower_d7 = Conv(data=tower_d7, num_filter=basefilter*num_d7_2, num_group=num_group, kernel=(7, 1), pad=(3, 0), name=('%s_tower' % name), suffix='_conv_2')
    # Branch 3:Conv11-Conv17-Conv71-Conv17-Conv71
    tower_q7 = Conv(data=data, num_filter=basefilter*num_q7_red, num_group=num_group_11, name=('%s_tower_1' % name), suffix='_conv')
    tower_q7 = Conv(data=tower_q7, num_filter=basefilter*num_q7_1, num_group=num_group, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_q7 = Conv(data=tower_q7, num_filter=basefilter*num_q7_2, num_group=num_group, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_2')
    tower_q7 = Conv(data=tower_q7, num_filter=basefilter*num_q7_3, num_group=num_group, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_3')
    tower_q7 = Conv(data=tower_q7, num_filter=basefilter*num_q7_4, num_group=num_group, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_4')
    # Branch4: Pooling-Conv11
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(data=pooling, num_filter=basefilter*proj, num_group=num_group_11, kernel=(1, 1), name=('%s_tower_2' %  name), suffix='_conv')
    # concat
    concat = mx.sym.Concat(*[tower_1x1, tower_d7, tower_q7, cproj], name='ch_concat_%s_chconcat' % name)
    return concat


# Second Downsample

# Field Change: (x-2)/2, original 18

def Inception7D_V3(data,
                basefilter=64, 
                num_filters=[], # Length-6
                num_group=1, num_group_11=1,
                pool='max',
                name=''):
    
    assert len(num_filters)==6
    
    num_3x3_red, num_3x3,\
                num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3 = tuple(num_filters)
    
    # Branch 1: Conv11-Conv33
    tower_3x3 = Conv(data=data, num_filter=basefilter*num_3x3_red, num_group=num_group_11, name=('%s_tower' % name), suffix='_conv')
    tower_3x3 = Conv(data=tower_3x3, num_filter=basefilter*num_3x3, num_group=num_group, kernel=(3, 3), pad=(0,0), stride=(2, 2), name=('%s_tower' % name), suffix='_conv_1')
    # Branch 2: Conv11-Conv17-Conv71-Conv33
    tower_d7_3x3 = Conv(data=data, num_filter=basefilter*num_d7_3x3_red, num_group=num_group_11,  name=('%s_tower_1' % name), suffix='_conv')
    tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=basefilter*num_d7_1, num_group=num_group, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=basefilter*num_d7_2, num_group=num_group, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_2')
    tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=basefilter*num_d7_3x3, num_group=num_group, kernel=(3, 3), stride=(2, 2), name=('%s_tower_1' % name), suffix='_conv_3')
    # Branch 3: Pool33
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    # concat
    concat = mx.sym.Concat(*[tower_3x3, tower_d7_3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat


# Doesn't change field

def Inception7E_V3(data,
                basefilter=64,
                num_filters=[], # Length-9
                num_group=1, num_group_11=1,
                pool='max', 
                name=''):
    
    assert len(num_filters)==9
    
    num_1x1, num_d3_red, num_d3_1, num_d3_2,\
                num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2, proj = tuple(num_filters)
    
    
    # Branch 1: Conv11
    tower_1x1 = Conv(data=data, num_filter=basefilter*num_1x1, num_group=num_group_11, kernel=(1, 1), name=('%s_conv' % name))
    # Branch 2: Conv11-Conv13-Conv31
    tower_d3 = Conv(data=data, num_filter=basefilter*num_d3_red, num_group=num_group_11, name=('%s_tower' % name), suffix='_conv')
    tower_d3_a = Conv(data=tower_d3, num_filter=basefilter*num_d3_1, num_group=num_group, kernel=(1, 3), pad=(0, 1), name=('%s_tower' % name), suffix='_mixed_conv')
    tower_d3_b = Conv(data=tower_d3, num_filter=basefilter*num_d3_2, num_group=num_group, kernel=(3, 1), pad=(1, 0), name=('%s_tower' % name), suffix='_mixed_conv_1')
    # Branch 3: Conv11-Conv33-Conv13-Conv31
    tower_3x3_d3 = Conv(data=data, num_filter=basefilter*num_3x3_d3_red, num_group=num_group_11, name=('%s_tower_1' % name), suffix='_conv')
    tower_3x3_d3 = Conv(data=tower_3x3_d3, num_filter=basefilter*num_3x3, num_group=num_group, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_3x3_d3_a = Conv(data=tower_3x3_d3, num_filter=basefilter*num_3x3_d3_1, num_group=num_group, kernel=(1, 3), pad=(0, 1), name=('%s_tower_1' % name), suffix='_mixed_conv')
    tower_3x3_d3_b = Conv(data=tower_3x3_d3, num_filter=basefilter*num_3x3_d3_2, num_group=num_group, kernel=(3, 1), pad=(1, 0), name=('%s_tower_1' % name), suffix='_mixed_conv_1')
    # Branch 4: Pool33-Conv11
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(data=pooling, num_filter=basefilter*proj, kernel=(1, 1), num_group=num_group_11, name=('%s_tower_2' %  name), suffix='_conv')
    # concat
    concat = mx.sym.Concat(*[tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj], name='ch_concat_%s_chconcat' % name)
    return concat



def get_symbol_V3(num_classes=1000, 
                  basefilter=16, num_group=1, num_group_11=1, num_group_stem=1,
                  lastout = 8,
                  dtype='float32', **kwargs):
    data = mx.sym.Variable(name="data")
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    # Stem Stage 1
    
    # 299 
    conv = Conv(data, basefilter*2, num_group=num_group_stem, kernel=(3, 3), stride=(2, 2), name="conv")
    # 149
    conv_1 = Conv(conv, basefilter*2, num_group=num_group_stem,  kernel=(3, 3), name="conv_1")
    # 147
    conv_2 = Conv(conv_1, basefilter*4, num_group=num_group_stem, kernel=(3, 3), pad=(1, 1), name="conv_2")
    # 147
    pool = mx.sym.Pooling(data=conv_2, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool")
    # 73
    # Stem Stage 2
    conv_3 = Conv(pool, basefilter*5, num_group=num_group_11, kernel=(1, 1), name="conv_3")
    conv_4 = Conv(conv_3, basefilter*12, num_group=num_group_stem, kernel=(3, 3), name="conv_4")
    # 71
    pool1 = mx.sym.Pooling(data=conv_4, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool1")
    # 35
    # Main Stage 1
    in3a = Inception7A_V3(pool1, 
                       basefilter=basefilter*1,
                       num_filters=[4,4,6,6,3,4,2],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="avg", name="mixed")
    in3b = Inception7A_V3(in3a, 
                       basefilter=basefilter*1,
                       num_filters=[4,4,6,6,3,4,2],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="avg", name="mixed_1")
    in3c = Inception7A_V3(in3b,
                       basefilter=basefilter*1,
                       num_filters=[4,4,6,6,3,4,2],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="avg", name="mixed_2")
    in3d = Inception7B_V3(in3c,
                       basefilter=basefilter*2,
                       num_filters=[12,2,3,3],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="max", name="mixed_3")
    # Main Stage2
    in4a = Inception7C_V3(in3d, 
                       basefilter=basefilter*2,
                       num_filters=[6,4,4,6,4,4,4,4,6,6],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="avg", name="mixed_4")
    in4b = Inception7C_V3(in4a, 
                       basefilter=basefilter*2,
                       num_filters=[6,5,5,6,5,5,5,5,6,6],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="avg", name="mixed_5")
    in4c = Inception7C_V3(in4b, 
                       basefilter=basefilter*2,
                       num_filters=[6,5,5,6,5,5,5,5,6,6],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="avg", name="mixed_6")
    in4d = Inception7C_V3(in4c,
                       basefilter=basefilter*2,
                       num_filters=[6,6,6,6,6,6,6,6,6,6],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="avg", name="mixed_7")
    in4e = Inception7D_V3(in4d, 
                       basefilter=basefilter*4,
                       num_filters=[3,5,3,3,3,3],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="max", name="mixed_8")
    # Main Stage3
    in5a = Inception7E_V3(in4e,
                       basefilter=basefilter*4,
                       num_filters=[5,6,6,6,7,6,6,6,3],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="avg", name="mixed_9")
    in5b = Inception7E_V3(in5a, 
                       basefilter=basefilter*4,
                       num_filters=[5,6,6,6,7,6,6,6,3],
                       num_group=num_group, num_group_11=num_group_11,
                       pool="max", name="mixed_10")
    # pool 
    pool = mx.sym.Pooling(data=in5b, kernel=(lastout, lastout), stride=(1, 1), pool_type="avg", name="global_pool") # last=8
    flatten = mx.sym.Flatten(data=pool, name="flatten")
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    softmax = mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

