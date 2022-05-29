# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None, data = tf.placeholder(tf.float32, shape = (None, 112, 112, 3), name = 'data')):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    minusscalar0_second = tf.constant(__weights_dict['minusscalar0_second']['value'], name='minusscalar0_second')

    mulscalar0_second = tf.constant(__weights_dict['mulscalar0_second']['value'], name='mulscalar0_second')
    minusscalar0    = data - minusscalar0_second
    mulscalar0      = minusscalar0 * mulscalar0_second
    conv0_pad       = tf.pad(mulscalar0, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv0           = convolution(conv0_pad, group=1, strides=[1, 1], padding='VALID', name='conv0')
    bn0             = batch_normalization(conv0, variance_epsilon=1.9999999494757503e-05, name='bn0')
    relu0           = prelu(bn0, name='relu0')
    stage1_unit1_bn1 = batch_normalization(relu0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn1')
    stage1_unit1_conv1sc = convolution(relu0, group=1, strides=[2, 2], padding='VALID', name='stage1_unit1_conv1sc')
    stage1_unit1_conv1_pad = tf.pad(stage1_unit1_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit1_conv1 = convolution(stage1_unit1_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit1_conv1')
    stage1_unit1_sc = batch_normalization(stage1_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_sc')
    stage1_unit1_bn2 = batch_normalization(stage1_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn2')
    stage1_unit1_relu1 = prelu(stage1_unit1_bn2, name='stage1_unit1_relu1')
    stage1_unit1_conv2_pad = tf.pad(stage1_unit1_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit1_conv2 = convolution(stage1_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage1_unit1_conv2')
    stage1_unit1_bn3 = batch_normalization(stage1_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn3')
    plus0           = stage1_unit1_bn3 + stage1_unit1_sc
    stage1_unit2_bn1 = batch_normalization(plus0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn1')
    stage1_unit2_conv1_pad = tf.pad(stage1_unit2_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit2_conv1 = convolution(stage1_unit2_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit2_conv1')
    stage1_unit2_bn2 = batch_normalization(stage1_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn2')
    stage1_unit2_relu1 = prelu(stage1_unit2_bn2, name='stage1_unit2_relu1')
    stage1_unit2_conv2_pad = tf.pad(stage1_unit2_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit2_conv2 = convolution(stage1_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit2_conv2')
    stage1_unit2_bn3 = batch_normalization(stage1_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn3')
    plus1           = stage1_unit2_bn3 + plus0
    stage1_unit3_bn1 = batch_normalization(plus1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn1')
    stage1_unit3_conv1_pad = tf.pad(stage1_unit3_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit3_conv1 = convolution(stage1_unit3_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit3_conv1')
    stage1_unit3_bn2 = batch_normalization(stage1_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn2')
    stage1_unit3_relu1 = prelu(stage1_unit3_bn2, name='stage1_unit3_relu1')
    stage1_unit3_conv2_pad = tf.pad(stage1_unit3_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit3_conv2 = convolution(stage1_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit3_conv2')
    stage1_unit3_bn3 = batch_normalization(stage1_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn3')
    plus2           = stage1_unit3_bn3 + plus1
    stage2_unit1_bn1 = batch_normalization(plus2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn1')
    stage2_unit1_conv1sc = convolution(plus2, group=1, strides=[2, 2], padding='VALID', name='stage2_unit1_conv1sc')
    stage2_unit1_conv1_pad = tf.pad(stage2_unit1_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit1_conv1 = convolution(stage2_unit1_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit1_conv1')
    stage2_unit1_sc = batch_normalization(stage2_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_sc')
    stage2_unit1_bn2 = batch_normalization(stage2_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn2')
    stage2_unit1_relu1 = prelu(stage2_unit1_bn2, name='stage2_unit1_relu1')
    stage2_unit1_conv2_pad = tf.pad(stage2_unit1_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit1_conv2 = convolution(stage2_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage2_unit1_conv2')
    stage2_unit1_bn3 = batch_normalization(stage2_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn3')
    plus3           = stage2_unit1_bn3 + stage2_unit1_sc
    stage2_unit2_bn1 = batch_normalization(plus3, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn1')
    stage2_unit2_conv1_pad = tf.pad(stage2_unit2_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit2_conv1 = convolution(stage2_unit2_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit2_conv1')
    stage2_unit2_bn2 = batch_normalization(stage2_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn2')
    stage2_unit2_relu1 = prelu(stage2_unit2_bn2, name='stage2_unit2_relu1')
    stage2_unit2_conv2_pad = tf.pad(stage2_unit2_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit2_conv2 = convolution(stage2_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit2_conv2')
    stage2_unit2_bn3 = batch_normalization(stage2_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn3')
    plus4           = stage2_unit2_bn3 + plus3
    stage2_unit3_bn1 = batch_normalization(plus4, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn1')
    stage2_unit3_conv1_pad = tf.pad(stage2_unit3_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit3_conv1 = convolution(stage2_unit3_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit3_conv1')
    stage2_unit3_bn2 = batch_normalization(stage2_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn2')
    stage2_unit3_relu1 = prelu(stage2_unit3_bn2, name='stage2_unit3_relu1')
    stage2_unit3_conv2_pad = tf.pad(stage2_unit3_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit3_conv2 = convolution(stage2_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit3_conv2')
    stage2_unit3_bn3 = batch_normalization(stage2_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn3')
    plus5           = stage2_unit3_bn3 + plus4
    stage2_unit4_bn1 = batch_normalization(plus5, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn1')
    stage2_unit4_conv1_pad = tf.pad(stage2_unit4_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit4_conv1 = convolution(stage2_unit4_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit4_conv1')
    stage2_unit4_bn2 = batch_normalization(stage2_unit4_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn2')
    stage2_unit4_relu1 = prelu(stage2_unit4_bn2, name='stage2_unit4_relu1')
    stage2_unit4_conv2_pad = tf.pad(stage2_unit4_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit4_conv2 = convolution(stage2_unit4_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit4_conv2')
    stage2_unit4_bn3 = batch_normalization(stage2_unit4_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn3')
    plus6           = stage2_unit4_bn3 + plus5
    stage3_unit1_bn1 = batch_normalization(plus6, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn1')
    stage3_unit1_conv1sc = convolution(plus6, group=1, strides=[2, 2], padding='VALID', name='stage3_unit1_conv1sc')
    stage3_unit1_conv1_pad = tf.pad(stage3_unit1_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit1_conv1 = convolution(stage3_unit1_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit1_conv1')
    stage3_unit1_sc = batch_normalization(stage3_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_sc')
    stage3_unit1_bn2 = batch_normalization(stage3_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn2')
    stage3_unit1_relu1 = prelu(stage3_unit1_bn2, name='stage3_unit1_relu1')
    stage3_unit1_conv2_pad = tf.pad(stage3_unit1_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit1_conv2 = convolution(stage3_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage3_unit1_conv2')
    stage3_unit1_bn3 = batch_normalization(stage3_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn3')
    plus7           = stage3_unit1_bn3 + stage3_unit1_sc
    stage3_unit2_bn1 = batch_normalization(plus7, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn1')
    stage3_unit2_conv1_pad = tf.pad(stage3_unit2_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit2_conv1 = convolution(stage3_unit2_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit2_conv1')
    stage3_unit2_bn2 = batch_normalization(stage3_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn2')
    stage3_unit2_relu1 = prelu(stage3_unit2_bn2, name='stage3_unit2_relu1')
    stage3_unit2_conv2_pad = tf.pad(stage3_unit2_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit2_conv2 = convolution(stage3_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit2_conv2')
    stage3_unit2_bn3 = batch_normalization(stage3_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn3')
    plus8           = stage3_unit2_bn3 + plus7
    stage3_unit3_bn1 = batch_normalization(plus8, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn1')
    stage3_unit3_conv1_pad = tf.pad(stage3_unit3_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit3_conv1 = convolution(stage3_unit3_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit3_conv1')
    stage3_unit3_bn2 = batch_normalization(stage3_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn2')
    stage3_unit3_relu1 = prelu(stage3_unit3_bn2, name='stage3_unit3_relu1')
    stage3_unit3_conv2_pad = tf.pad(stage3_unit3_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit3_conv2 = convolution(stage3_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit3_conv2')
    stage3_unit3_bn3 = batch_normalization(stage3_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn3')
    plus9           = stage3_unit3_bn3 + plus8
    stage3_unit4_bn1 = batch_normalization(plus9, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn1')
    stage3_unit4_conv1_pad = tf.pad(stage3_unit4_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit4_conv1 = convolution(stage3_unit4_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit4_conv1')
    stage3_unit4_bn2 = batch_normalization(stage3_unit4_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn2')
    stage3_unit4_relu1 = prelu(stage3_unit4_bn2, name='stage3_unit4_relu1')
    stage3_unit4_conv2_pad = tf.pad(stage3_unit4_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit4_conv2 = convolution(stage3_unit4_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit4_conv2')
    stage3_unit4_bn3 = batch_normalization(stage3_unit4_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn3')
    plus10          = stage3_unit4_bn3 + plus9
    stage3_unit5_bn1 = batch_normalization(plus10, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn1')
    stage3_unit5_conv1_pad = tf.pad(stage3_unit5_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit5_conv1 = convolution(stage3_unit5_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit5_conv1')
    stage3_unit5_bn2 = batch_normalization(stage3_unit5_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn2')
    stage3_unit5_relu1 = prelu(stage3_unit5_bn2, name='stage3_unit5_relu1')
    stage3_unit5_conv2_pad = tf.pad(stage3_unit5_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit5_conv2 = convolution(stage3_unit5_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit5_conv2')
    stage3_unit5_bn3 = batch_normalization(stage3_unit5_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn3')
    plus11          = stage3_unit5_bn3 + plus10
    stage3_unit6_bn1 = batch_normalization(plus11, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn1')
    stage3_unit6_conv1_pad = tf.pad(stage3_unit6_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit6_conv1 = convolution(stage3_unit6_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit6_conv1')
    stage3_unit6_bn2 = batch_normalization(stage3_unit6_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn2')
    stage3_unit6_relu1 = prelu(stage3_unit6_bn2, name='stage3_unit6_relu1')
    stage3_unit6_conv2_pad = tf.pad(stage3_unit6_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit6_conv2 = convolution(stage3_unit6_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit6_conv2')
    stage3_unit6_bn3 = batch_normalization(stage3_unit6_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn3')
    plus12          = stage3_unit6_bn3 + plus11
    stage3_unit7_bn1 = batch_normalization(plus12, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn1')
    stage3_unit7_conv1_pad = tf.pad(stage3_unit7_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit7_conv1 = convolution(stage3_unit7_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit7_conv1')
    stage3_unit7_bn2 = batch_normalization(stage3_unit7_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn2')
    stage3_unit7_relu1 = prelu(stage3_unit7_bn2, name='stage3_unit7_relu1')
    stage3_unit7_conv2_pad = tf.pad(stage3_unit7_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit7_conv2 = convolution(stage3_unit7_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit7_conv2')
    stage3_unit7_bn3 = batch_normalization(stage3_unit7_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn3')
    plus13          = stage3_unit7_bn3 + plus12
    stage3_unit8_bn1 = batch_normalization(plus13, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn1')
    stage3_unit8_conv1_pad = tf.pad(stage3_unit8_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit8_conv1 = convolution(stage3_unit8_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit8_conv1')
    stage3_unit8_bn2 = batch_normalization(stage3_unit8_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn2')
    stage3_unit8_relu1 = prelu(stage3_unit8_bn2, name='stage3_unit8_relu1')
    stage3_unit8_conv2_pad = tf.pad(stage3_unit8_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit8_conv2 = convolution(stage3_unit8_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit8_conv2')
    stage3_unit8_bn3 = batch_normalization(stage3_unit8_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn3')
    plus14          = stage3_unit8_bn3 + plus13
    stage3_unit9_bn1 = batch_normalization(plus14, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn1')
    stage3_unit9_conv1_pad = tf.pad(stage3_unit9_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit9_conv1 = convolution(stage3_unit9_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit9_conv1')
    stage3_unit9_bn2 = batch_normalization(stage3_unit9_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn2')
    stage3_unit9_relu1 = prelu(stage3_unit9_bn2, name='stage3_unit9_relu1')
    stage3_unit9_conv2_pad = tf.pad(stage3_unit9_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit9_conv2 = convolution(stage3_unit9_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit9_conv2')
    stage3_unit9_bn3 = batch_normalization(stage3_unit9_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn3')
    plus15          = stage3_unit9_bn3 + plus14
    stage3_unit10_bn1 = batch_normalization(plus15, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn1')
    stage3_unit10_conv1_pad = tf.pad(stage3_unit10_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit10_conv1 = convolution(stage3_unit10_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit10_conv1')
    stage3_unit10_bn2 = batch_normalization(stage3_unit10_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn2')
    stage3_unit10_relu1 = prelu(stage3_unit10_bn2, name='stage3_unit10_relu1')
    stage3_unit10_conv2_pad = tf.pad(stage3_unit10_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit10_conv2 = convolution(stage3_unit10_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit10_conv2')
    stage3_unit10_bn3 = batch_normalization(stage3_unit10_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn3')
    plus16          = stage3_unit10_bn3 + plus15
    stage3_unit11_bn1 = batch_normalization(plus16, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn1')
    stage3_unit11_conv1_pad = tf.pad(stage3_unit11_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit11_conv1 = convolution(stage3_unit11_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit11_conv1')
    stage3_unit11_bn2 = batch_normalization(stage3_unit11_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn2')
    stage3_unit11_relu1 = prelu(stage3_unit11_bn2, name='stage3_unit11_relu1')
    stage3_unit11_conv2_pad = tf.pad(stage3_unit11_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit11_conv2 = convolution(stage3_unit11_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit11_conv2')
    stage3_unit11_bn3 = batch_normalization(stage3_unit11_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn3')
    plus17          = stage3_unit11_bn3 + plus16
    stage3_unit12_bn1 = batch_normalization(plus17, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn1')
    stage3_unit12_conv1_pad = tf.pad(stage3_unit12_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit12_conv1 = convolution(stage3_unit12_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit12_conv1')
    stage3_unit12_bn2 = batch_normalization(stage3_unit12_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn2')
    stage3_unit12_relu1 = prelu(stage3_unit12_bn2, name='stage3_unit12_relu1')
    stage3_unit12_conv2_pad = tf.pad(stage3_unit12_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit12_conv2 = convolution(stage3_unit12_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit12_conv2')
    stage3_unit12_bn3 = batch_normalization(stage3_unit12_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn3')
    plus18          = stage3_unit12_bn3 + plus17
    stage3_unit13_bn1 = batch_normalization(plus18, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn1')
    stage3_unit13_conv1_pad = tf.pad(stage3_unit13_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit13_conv1 = convolution(stage3_unit13_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit13_conv1')
    stage3_unit13_bn2 = batch_normalization(stage3_unit13_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn2')
    stage3_unit13_relu1 = prelu(stage3_unit13_bn2, name='stage3_unit13_relu1')
    stage3_unit13_conv2_pad = tf.pad(stage3_unit13_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit13_conv2 = convolution(stage3_unit13_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit13_conv2')
    stage3_unit13_bn3 = batch_normalization(stage3_unit13_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn3')
    plus19          = stage3_unit13_bn3 + plus18
    stage3_unit14_bn1 = batch_normalization(plus19, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn1')
    stage3_unit14_conv1_pad = tf.pad(stage3_unit14_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit14_conv1 = convolution(stage3_unit14_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit14_conv1')
    stage3_unit14_bn2 = batch_normalization(stage3_unit14_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn2')
    stage3_unit14_relu1 = prelu(stage3_unit14_bn2, name='stage3_unit14_relu1')
    stage3_unit14_conv2_pad = tf.pad(stage3_unit14_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit14_conv2 = convolution(stage3_unit14_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit14_conv2')
    stage3_unit14_bn3 = batch_normalization(stage3_unit14_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn3')
    plus20          = stage3_unit14_bn3 + plus19
    stage4_unit1_bn1 = batch_normalization(plus20, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn1')
    stage4_unit1_conv1sc = convolution(plus20, group=1, strides=[2, 2], padding='VALID', name='stage4_unit1_conv1sc')
    stage4_unit1_conv1_pad = tf.pad(stage4_unit1_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit1_conv1 = convolution(stage4_unit1_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit1_conv1')
    stage4_unit1_sc = batch_normalization(stage4_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_sc')
    stage4_unit1_bn2 = batch_normalization(stage4_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn2')
    stage4_unit1_relu1 = prelu(stage4_unit1_bn2, name='stage4_unit1_relu1')
    stage4_unit1_conv2_pad = tf.pad(stage4_unit1_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit1_conv2 = convolution(stage4_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage4_unit1_conv2')
    stage4_unit1_bn3 = batch_normalization(stage4_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn3')
    plus21          = stage4_unit1_bn3 + stage4_unit1_sc
    stage4_unit2_bn1 = batch_normalization(plus21, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn1')
    stage4_unit2_conv1_pad = tf.pad(stage4_unit2_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit2_conv1 = convolution(stage4_unit2_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit2_conv1')
    stage4_unit2_bn2 = batch_normalization(stage4_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn2')
    stage4_unit2_relu1 = prelu(stage4_unit2_bn2, name='stage4_unit2_relu1')
    stage4_unit2_conv2_pad = tf.pad(stage4_unit2_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit2_conv2 = convolution(stage4_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit2_conv2')
    stage4_unit2_bn3 = batch_normalization(stage4_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn3')
    plus22          = stage4_unit2_bn3 + plus21
    stage4_unit3_bn1 = batch_normalization(plus22, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn1')
    stage4_unit3_conv1_pad = tf.pad(stage4_unit3_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit3_conv1 = convolution(stage4_unit3_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit3_conv1')
    stage4_unit3_bn2 = batch_normalization(stage4_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn2')
    stage4_unit3_relu1 = prelu(stage4_unit3_bn2, name='stage4_unit3_relu1')
    stage4_unit3_conv2_pad = tf.pad(stage4_unit3_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit3_conv2 = convolution(stage4_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit3_conv2')
    stage4_unit3_bn3 = batch_normalization(stage4_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn3')
    plus23          = stage4_unit3_bn3 + plus22
    bn1             = batch_normalization(plus23, variance_epsilon=1.9999999494757503e-05, name='bn1')
    pre_fc1_flatten = tf.contrib.layers.flatten(bn1)
    pre_fc1         = tf.layers.dense(pre_fc1_flatten, 512, kernel_initializer = tf.constant_initializer(__weights_dict['pre_fc1']['weights']), bias_initializer = tf.constant_initializer(__weights_dict['pre_fc1']['bias']), use_bias = True,reuse=tf.AUTO_REUSE)
    fc1             = batch_normalization(pre_fc1, variance_epsilon=1.9999999494757503e-05, name='fc1')
    return data, fc1, [stage2_unit1_bn1,stage3_unit1_bn1,stage4_unit1_bn1,pre_fc1_flatten]


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer

def prelu(input, name):
    gamma = tf.Variable(__weights_dict[name]['gamma'], name=name + "_gamma", trainable=is_train)
    return tf.maximum(0.0, input) + gamma * tf.minimum(0.0, input)
    

def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)

