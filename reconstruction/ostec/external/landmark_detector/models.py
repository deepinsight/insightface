import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

# custom layers

def deconv_layer(net, up_scale, n_channel, method='transpose'):
    nh = tf.shape(net)[-3] * up_scale
    nw = tf.shape(net)[-2] * up_scale

    if method == 'transpose':
        net = slim.conv2d_transpose(net, n_channel, (up_scale, up_scale), (
            up_scale, up_scale), activation_fn=None, padding='VALID')
    elif method == 'transpose+conv':
        net = slim.conv2d_transpose(net, n_channel, (up_scale, up_scale), (
            up_scale, up_scale), activation_fn=None, padding='VALID')
        net = slim.conv2d(net, n_channel, (3, 3), (1, 1))
    elif method == 'transpose+conv+relu':
        net = slim.conv2d_transpose(net, n_channel, (up_scale, up_scale), (
            up_scale, up_scale), padding='VALID')
        net = slim.conv2d(net, n_channel, (3, 3), (1, 1))
    elif method == 'bilinear':
        net = tf.image.resize_images(net, [nh, nw])
    else:
        raise Exception('Unrecognised Deconvolution Method: %s' % method)

    return net


# arg scopes
def hourglass_arg_scope_torch(weight_decay=0.0001,
                              batch_norm_decay=0.997,
                              batch_norm_epsilon=1e-5,
                              batch_norm_scale=True):
    """Defines the default ResNet arg scope.
  Args:
    is_training: Whether or not we are training the parameters in the batch
      normalization layers of the model.
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=None,
            normalizer_fn=None,
            normalizer_params=None):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def hourglass_arg_scope_tf(weight_decay=0.0001,
                           batch_norm_decay=0.997,
                           batch_norm_epsilon=1e-5,
                           batch_norm_scale=True):
    """Defines the default ResNet arg scope.
  Args:
    is_training: Whether or not we are training the parameters in the batch
      normalization layers of the model.
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


# bottleneck_inception_SE
def bottleneck_inception_SE_module(
        inputs,
        out_channel=256,
        res=None,
        scope='inception_block'):

    min_channel = out_channel // 8
    with tf.variable_scope(scope):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs, min_channel * 3,
                                   [1, 1], scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs, min_channel *
                                   3 / 2, [1, 1], scope='Conv2d_1x1')
            branch_1 = slim.conv2d(
                branch_1, min_channel * 3, [3, 3], scope='Conv2d_3x3')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs, min_channel //
                                   3, [1, 1], scope='Conv2d_1x1')
            branch_2 = slim.conv2d(
                branch_2, min_channel, [3, 3], scope='Conv2d_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(inputs, [3, 3], 1, scope='MaxPool_3x3')
            branch_3 = slim.conv2d(
                branch_3, min_channel, [1, 1], scope='Conv2d_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        se_branch = tf.reduce_mean(net, axis=[1, 2])
        se_branch = slim.fully_connected(se_branch, out_channel // 16)
        se_branch = slim.fully_connected(
            se_branch, out_channel, activation_fn=tf.sigmoid)

        net = net * se_branch[:,None,None,:]

        if res:
            inputs = slim.conv2d(inputs, res, (1, 1),
                                 scope='bn_res'.format(scope))

        net += inputs

    return net


# bottle neck modules
def bottleneck_inception_module(
        inputs,
        out_channel=256,
        res=None,
        scope='inception_block'):

    min_channel = out_channel // 8
    with tf.variable_scope(scope):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs, min_channel * 3,
                                   [1, 1], scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs, min_channel *
                                   3 / 2, [1, 1], scope='Conv2d_1x1')
            branch_1 = slim.conv2d(
                branch_1, min_channel * 3, [3, 3], scope='Conv2d_3x3')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs, min_channel //
                                   3, [1, 1], scope='Conv2d_1x1')
            branch_2 = slim.conv2d(
                branch_2, min_channel, [3, 3], scope='Conv2d_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(inputs, [3, 3], 1, scope='MaxPool_3x3')
            branch_3 = slim.conv2d(
                branch_3, min_channel, [1, 1], scope='Conv2d_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        if res:
            inputs = slim.conv2d(inputs, res, (1, 1),
                                 scope='bn_res'.format(scope))

        net += inputs

    return net


def bottleneck_module(inputs, out_channel=256, res=None, scope=''):

    with tf.variable_scope(scope):
        net = slim.stack(inputs, slim.conv2d, [
                         (out_channel // 2, [1, 1]), (out_channel // 2, [3, 3]), (out_channel, [1, 1])], scope='conv')
        if res:
            inputs = slim.conv2d(inputs, res, (1, 1),
                                 scope='bn_res'.format(scope))
        net += inputs

        return net


# recursive hourglass definition
def hourglass_module(inputs, depth=0, deconv='bilinear', bottleneck='bottleneck'):

    bm_fn = globals()['%s_module' % bottleneck]

    with tf.variable_scope('depth_{}'.format(depth)):
        # buttom up layers
        net = slim.max_pool2d(inputs, [2, 2], scope='pool')
        net = slim.stack(net, bm_fn, [
                         (256, None), (256, None), (256, None)], scope='buttom_up')

        # connecting layers
        if depth > 0:
            net = hourglass_module(net, depth=depth - 1, deconv=deconv)
        else:
            net = bm_fn(
                net, out_channel=512, res=512, scope='connecting')

        # top down layers
        net = bm_fn(net, out_channel=512,
                    res=512, scope='top_down')
        net = deconv_layer(net, 2, 512, method=deconv)
        # residual layers
        net += slim.stack(inputs, bm_fn,
                          [(256, None), (256, None), (512, 512)], scope='res')

        return net


def hourglass(inputs,
              scale=1,
              regression_channels=2,
              classification_channels=22,
              deconv='bilinear',
              bottleneck='bottleneck'):
    """Defines a lightweight resnet based model for dense estimation tasks.
    Args:
      inputs: A `Tensor` with dimensions [num_batches, height, width, depth].
      scale: A scalar which denotes the factor to subsample the current image.
      output_channels: The number of output channels. E.g., for human pose
        estimation this equals 13 channels.
    Returns:
      A `Tensor` of dimensions [num_batches, height, width, output_channels]."""

    out_shape = tf.shape(inputs)[1:3]

    if scale > 1:
        inputs = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)))
        inputs = slim.layers.avg_pool2d(
            inputs, (3, 3), (scale, scale), padding='VALID')

    output_channels = regression_channels + classification_channels

    with slim.arg_scope(hourglass_arg_scope_tf()):
        # D1
        net = slim.conv2d(inputs, 64, (7, 7), 2, scope='conv1')
        net = bottleneck_module(net, out_channel=128,
                                res=128, scope='bottleneck1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # D2
        net = slim.stack(net, bottleneck_module, [
                         (128, None), (128, None), (256, 256)], scope='conv2')

        # hourglasses (D3,D4,D5)
        with tf.variable_scope('hourglass'):
            net = hourglass_module(
                net, depth=4, deconv=deconv, bottleneck=bottleneck)

        # final layers (D6, D7)
        net = slim.stack(net, slim.conv2d, [(512, [1, 1]), (256, [1, 1]),
                                            (output_channels, [1, 1])
                                            ], scope='conv3')

        net = deconv_layer(net, 4, output_channels, method=deconv)
        net = slim.conv2d(net, output_channels, 1, scope='conv_last')

    regression = slim.conv2d(
        net, regression_channels, 1, activation_fn=None
    ) if regression_channels else None

    logits = slim.conv2d(
        net, classification_channels, 1, activation_fn=None
    ) if classification_channels else None

    return regression, logits


def StackedHourglassTorch(inputs, out_channels=16, deconv='bilinear'):
    net = inputs
    with tf.name_scope('nn.Sequential'):
        with tf.name_scope('nn.Sequential'):
            net = tf.pad(net, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
            net = slim.conv2d(net, 64, (7, 7), (2, 2),
                              activation_fn=None, padding='VALID')
            net = slim.batch_norm(net)
            net = slim.nn.relu(net)
            with tf.name_scope('nn.Sequential'):
                with tf.name_scope('nn.ConcatTable'):
                    net0 = net
                    with tf.name_scope('nn.Sequential'):
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 64, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 64, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                    net1 = net
                    with tf.name_scope('nn.Sequential'):
                        net1 = tf.pad(net1, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net1 = slim.conv2d(
                            net1, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                net = tf.add_n([net0, net1])
            net = tf.pad(net, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
            net = slim.max_pool2d(net, (2, 2), (2, 2))
            with tf.name_scope('nn.Sequential'):
                with tf.name_scope('nn.ConcatTable'):
                    net0 = net
                    with tf.name_scope('nn.Sequential'):
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 64, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 64, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                    net1 = net
                net = tf.add_n([net0, net1])
            with tf.name_scope('nn.Sequential'):
                with tf.name_scope('nn.ConcatTable'):
                    net0 = net
                    with tf.name_scope('nn.Sequential'):
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 64, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 64, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                    net1 = net
                net = tf.add_n([net0, net1])
            with tf.name_scope('nn.Sequential'):
                with tf.name_scope('nn.ConcatTable'):
                    net0 = net
                    with tf.name_scope('nn.Sequential'):
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                        net0 = slim.batch_norm(net0)
                        net0 = slim.nn.relu(net0)
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.conv2d(
                            net0, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                    net1 = net
                    with tf.name_scope('nn.Sequential'):
                        net1 = tf.pad(net1, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net1 = slim.conv2d(
                            net1, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                net = tf.add_n([net0, net1])
            with tf.name_scope('nn.Sequential'):
                with tf.name_scope('nn.ConcatTable'):
                    net0 = net
                    with tf.name_scope('nn.Sequential'):
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = slim.max_pool2d(net0, (2, 2), (2, 2))
                        with tf.name_scope('nn.Sequential'):
                            with tf.name_scope('nn.ConcatTable'):
                                net00 = net0
                                with tf.name_scope('nn.Sequential'):
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                net01 = net0
                            net0 = tf.add_n([net00, net01])
                        with tf.name_scope('nn.Sequential'):
                            with tf.name_scope('nn.ConcatTable'):
                                net00 = net0
                                with tf.name_scope('nn.Sequential'):
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                net01 = net0
                            net0 = tf.add_n([net00, net01])
                        with tf.name_scope('nn.Sequential'):
                            with tf.name_scope('nn.ConcatTable'):
                                net00 = net0
                                with tf.name_scope('nn.Sequential'):
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                    net00 = slim.batch_norm(net00)
                                    net00 = slim.nn.relu(net00)
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net00 = slim.conv2d(
                                        net00, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                net01 = net0
                            net0 = tf.add_n([net00, net01])
                        with tf.name_scope('nn.Sequential'):
                            with tf.name_scope('nn.ConcatTable'):
                                net00 = net0
                                with tf.name_scope('nn.Sequential'):
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net00 = slim.max_pool2d(
                                        net00, (2, 2), (2, 2))
                                    with tf.name_scope('nn.Sequential'):
                                        with tf.name_scope('nn.ConcatTable'):
                                            net000 = net00
                                            with tf.name_scope('nn.Sequential'):
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                            net001 = net00
                                        net00 = tf.add_n([net000, net001])
                                    with tf.name_scope('nn.Sequential'):
                                        with tf.name_scope('nn.ConcatTable'):
                                            net000 = net00
                                            with tf.name_scope('nn.Sequential'):
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                            net001 = net00
                                        net00 = tf.add_n([net000, net001])
                                    with tf.name_scope('nn.Sequential'):
                                        with tf.name_scope('nn.ConcatTable'):
                                            net000 = net00
                                            with tf.name_scope('nn.Sequential'):
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                net000 = slim.batch_norm(
                                                    net000)
                                                net000 = slim.nn.relu(net000)
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net000 = slim.conv2d(
                                                    net000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                            net001 = net00
                                        net00 = tf.add_n([net000, net001])
                                    with tf.name_scope('nn.Sequential'):
                                        with tf.name_scope('nn.ConcatTable'):
                                            net000 = net00
                                            with tf.name_scope('nn.Sequential'):
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net000 = slim.max_pool2d(
                                                    net000, (2, 2), (2, 2))
                                                with tf.name_scope('nn.Sequential'):
                                                    with tf.name_scope('nn.ConcatTable'):
                                                        net0000 = net000
                                                        with tf.name_scope('nn.Sequential'):
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                        net0001 = net000
                                                    net000 = tf.add_n(
                                                        [net0000, net0001])
                                                with tf.name_scope('nn.Sequential'):
                                                    with tf.name_scope('nn.ConcatTable'):
                                                        net0000 = net000
                                                        with tf.name_scope('nn.Sequential'):
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                        net0001 = net000
                                                    net000 = tf.add_n(
                                                        [net0000, net0001])
                                                with tf.name_scope('nn.Sequential'):
                                                    with tf.name_scope('nn.ConcatTable'):
                                                        net0000 = net000
                                                        with tf.name_scope('nn.Sequential'):
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                            net0000 = slim.batch_norm(
                                                                net0000)
                                                            net0000 = slim.nn.relu(
                                                                net0000)
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0000 = slim.conv2d(
                                                                net0000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                        net0001 = net000
                                                    net000 = tf.add_n(
                                                        [net0000, net0001])
                                                with tf.name_scope('nn.Sequential'):
                                                    with tf.name_scope('nn.ConcatTable'):
                                                        net0000 = net000
                                                        with tf.name_scope('nn.Sequential'):
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0000 = slim.max_pool2d(
                                                                net0000, (2, 2), (2, 2))
                                                            with tf.name_scope('nn.Sequential'):
                                                                with tf.name_scope('nn.ConcatTable'):
                                                                    net00000 = net0000
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                    net00001 = net0000
                                                                net0000 = tf.add_n(
                                                                    [net00000, net00001])
                                                            with tf.name_scope('nn.Sequential'):
                                                                with tf.name_scope('nn.ConcatTable'):
                                                                    net00000 = net0000
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                    net00001 = net0000
                                                                net0000 = tf.add_n(
                                                                    [net00000, net00001])
                                                            with tf.name_scope('nn.Sequential'):
                                                                with tf.name_scope('nn.ConcatTable'):
                                                                    net00000 = net0000
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                    net00001 = net0000
                                                                net0000 = tf.add_n(
                                                                    [net00000, net00001])
                                                            with tf.name_scope('nn.Sequential'):
                                                                with tf.name_scope('nn.ConcatTable'):
                                                                    net00000 = net0000
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                    net00001 = net0000
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00001 = tf.pad(net00001, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00001 = slim.conv2d(
                                                                            net00001, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                net0000 = tf.add_n(
                                                                    [net00000, net00001])
                                                            with tf.name_scope('nn.Sequential'):
                                                                with tf.name_scope('nn.ConcatTable'):
                                                                    net00000 = net0000
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00000 = slim.batch_norm(
                                                                            net00000)
                                                                        net00000 = slim.nn.relu(
                                                                            net00000)
                                                                        net00000 = tf.pad(net00000, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00000 = slim.conv2d(
                                                                            net00000, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                    net00001 = net0000
                                                                net0000 = tf.add_n(
                                                                    [net00000, net00001])
                                                            net0000 = tf.pad(net0000, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0000 = deconv_layer(
                                                                net0000, 2, 512, method=deconv)
                                                        net0001 = net000
                                                        with tf.name_scope('nn.Sequential'):
                                                            with tf.name_scope('nn.Sequential'):
                                                                with tf.name_scope('nn.ConcatTable'):
                                                                    net00010 = net0001
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                    net00011 = net0001
                                                                net0001 = tf.add_n(
                                                                    [net00010, net00011])
                                                            with tf.name_scope('nn.Sequential'):
                                                                with tf.name_scope('nn.ConcatTable'):
                                                                    net00010 = net0001
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                    net00011 = net0001
                                                                net0001 = tf.add_n(
                                                                    [net00010, net00011])
                                                            with tf.name_scope('nn.Sequential'):
                                                                with tf.name_scope('nn.ConcatTable'):
                                                                    net00010 = net0001
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                        net00010 = slim.batch_norm(
                                                                            net00010)
                                                                        net00010 = slim.nn.relu(
                                                                            net00010)
                                                                        net00010 = tf.pad(net00010, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00010 = slim.conv2d(
                                                                            net00010, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                    net00011 = net0001
                                                                    with tf.name_scope('nn.Sequential'):
                                                                        net00011 = tf.pad(net00011, np.array(
                                                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                        net00011 = slim.conv2d(
                                                                            net00011, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                net0001 = tf.add_n(
                                                                    [net00010, net00011])
                                                    net000 = tf.add_n(
                                                        [net0000, net0001])
                                                    with tf.name_scope('nn.Sequential'):
                                                        with tf.name_scope('nn.ConcatTable'):
                                                            net0000 = net000
                                                            with tf.name_scope('nn.Sequential'):
                                                                net0000 = slim.batch_norm(
                                                                    net0000)
                                                                net0000 = slim.nn.relu(
                                                                    net0000)
                                                                net0000 = tf.pad(net0000, np.array(
                                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                net0000 = slim.conv2d(
                                                                    net0000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                                net0000 = slim.batch_norm(
                                                                    net0000)
                                                                net0000 = slim.nn.relu(
                                                                    net0000)
                                                                net0000 = tf.pad(net0000, np.array(
                                                                    [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                                net0000 = slim.conv2d(
                                                                    net0000, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                                net0000 = slim.batch_norm(
                                                                    net0000)
                                                                net0000 = slim.nn.relu(
                                                                    net0000)
                                                                net0000 = tf.pad(net0000, np.array(
                                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                                net0000 = slim.conv2d(
                                                                    net0000, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                            net0001 = net000
                                                        net000 = tf.add_n(
                                                            [net0000, net0001])
                                                net000 = tf.pad(net000, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net000 = deconv_layer(
                                                    net000, 2, 512, method=deconv)
                                            net001 = net00
                                            with tf.name_scope('nn.Sequential'):
                                                with tf.name_scope('nn.Sequential'):
                                                    with tf.name_scope('nn.ConcatTable'):
                                                        net0010 = net001
                                                        with tf.name_scope('nn.Sequential'):
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                        net0011 = net001
                                                    net001 = tf.add_n(
                                                        [net0010, net0011])
                                                with tf.name_scope('nn.Sequential'):
                                                    with tf.name_scope('nn.ConcatTable'):
                                                        net0010 = net001
                                                        with tf.name_scope('nn.Sequential'):
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                        net0011 = net001
                                                    net001 = tf.add_n(
                                                        [net0010, net0011])
                                                with tf.name_scope('nn.Sequential'):
                                                    with tf.name_scope('nn.ConcatTable'):
                                                        net0010 = net001
                                                        with tf.name_scope('nn.Sequential'):
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                            net0010 = slim.batch_norm(
                                                                net0010)
                                                            net0010 = slim.nn.relu(
                                                                net0010)
                                                            net0010 = tf.pad(net0010, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0010 = slim.conv2d(
                                                                net0010, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                        net0011 = net001
                                                        with tf.name_scope('nn.Sequential'):
                                                            net0011 = tf.pad(net0011, np.array(
                                                                [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                            net0011 = slim.conv2d(
                                                                net0011, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                    net001 = tf.add_n(
                                                        [net0010, net0011])
                                        net00 = tf.add_n([net000, net001])
                                        with tf.name_scope('nn.Sequential'):
                                            with tf.name_scope('nn.ConcatTable'):
                                                net000 = net00
                                                with tf.name_scope('nn.Sequential'):
                                                    net000 = slim.batch_norm(
                                                        net000)
                                                    net000 = slim.nn.relu(
                                                        net000)
                                                    net000 = tf.pad(net000, np.array(
                                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                    net000 = slim.conv2d(
                                                        net000, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                    net000 = slim.batch_norm(
                                                        net000)
                                                    net000 = slim.nn.relu(
                                                        net000)
                                                    net000 = tf.pad(net000, np.array(
                                                        [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                    net000 = slim.conv2d(
                                                        net000, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                    net000 = slim.batch_norm(
                                                        net000)
                                                    net000 = slim.nn.relu(
                                                        net000)
                                                    net000 = tf.pad(net000, np.array(
                                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                    net000 = slim.conv2d(
                                                        net000, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                net001 = net00
                                            net00 = tf.add_n([net000, net001])
                                    net00 = tf.pad(net00, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net00 = deconv_layer(
                                        net00, 2, 512, method=deconv)
                                net01 = net0
                                with tf.name_scope('nn.Sequential'):
                                    with tf.name_scope('nn.Sequential'):
                                        with tf.name_scope('nn.ConcatTable'):
                                            net010 = net01
                                            with tf.name_scope('nn.Sequential'):
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                            net011 = net01
                                        net01 = tf.add_n([net010, net011])
                                    with tf.name_scope('nn.Sequential'):
                                        with tf.name_scope('nn.ConcatTable'):
                                            net010 = net01
                                            with tf.name_scope('nn.Sequential'):
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                            net011 = net01
                                        net01 = tf.add_n([net010, net011])
                                    with tf.name_scope('nn.Sequential'):
                                        with tf.name_scope('nn.ConcatTable'):
                                            net010 = net01
                                            with tf.name_scope('nn.Sequential'):
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                                net010 = slim.batch_norm(
                                                    net010)
                                                net010 = slim.nn.relu(net010)
                                                net010 = tf.pad(net010, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net010 = slim.conv2d(
                                                    net010, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                            net011 = net01
                                            with tf.name_scope('nn.Sequential'):
                                                net011 = tf.pad(net011, np.array(
                                                    [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                                net011 = slim.conv2d(
                                                    net011, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                        net01 = tf.add_n([net010, net011])
                            net0 = tf.add_n([net00, net01])
                            with tf.name_scope('nn.Sequential'):
                                with tf.name_scope('nn.ConcatTable'):
                                    net00 = net0
                                    with tf.name_scope('nn.Sequential'):
                                        net00 = slim.batch_norm(net00)
                                        net00 = slim.nn.relu(net00)
                                        net00 = tf.pad(net00, np.array(
                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                        net00 = slim.conv2d(
                                            net00, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                        net00 = slim.batch_norm(net00)
                                        net00 = slim.nn.relu(net00)
                                        net00 = tf.pad(net00, np.array(
                                            [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                        net00 = slim.conv2d(
                                            net00, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                        net00 = slim.batch_norm(net00)
                                        net00 = slim.nn.relu(net00)
                                        net00 = tf.pad(net00, np.array(
                                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                        net00 = slim.conv2d(
                                            net00, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                    net01 = net0
                                net0 = tf.add_n([net00, net01])
                        net0 = tf.pad(net0, np.array(
                            [[0, 0], [0, 0], [0, 0], [0, 0]]))
                        net0 = deconv_layer(net0, 2, 512, method=deconv)

                    net1 = net
                    with tf.name_scope('nn.Sequential'):
                        with tf.name_scope('nn.Sequential'):
                            with tf.name_scope('nn.ConcatTable'):
                                net10 = net1
                                with tf.name_scope('nn.Sequential'):
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                net11 = net1
                            net1 = tf.add_n([net10, net11])
                        with tf.name_scope('nn.Sequential'):
                            with tf.name_scope('nn.ConcatTable'):
                                net10 = net1
                                with tf.name_scope('nn.Sequential'):
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 128, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 128, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                net11 = net1
                            net1 = tf.add_n([net10, net11])
                        with tf.name_scope('nn.Sequential'):
                            with tf.name_scope('nn.ConcatTable'):
                                net10 = net1
                                with tf.name_scope('nn.Sequential'):
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 256, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [1, 1], [1, 1], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 256, (3, 3), (1, 1), activation_fn=None, padding='VALID')
                                    net10 = slim.batch_norm(net10)
                                    net10 = slim.nn.relu(net10)
                                    net10 = tf.pad(net10, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net10 = slim.conv2d(
                                        net10, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                                net11 = net1
                                with tf.name_scope('nn.Sequential'):
                                    net11 = tf.pad(net11, np.array(
                                        [[0, 0], [0, 0], [0, 0], [0, 0]]))
                                    net11 = slim.conv2d(
                                        net11, 512, (1, 1), (1, 1), activation_fn=None, padding='VALID')
                            net1 = tf.add_n([net10, net11])
                net = tf.add_n([net0, net1])
            net = tf.pad(net, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
            net = slim.conv2d(net, 512, (1, 1), (1, 1),
                              activation_fn=None, padding='VALID')
            net = slim.batch_norm(net)
            net = slim.nn.relu(net)
            net = tf.pad(net, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
            net = slim.conv2d(net, 256, (1, 1), (1, 1),
                              activation_fn=None, padding='VALID')
            net = slim.batch_norm(net)
            net = slim.nn.relu(net)
            net = tf.pad(net, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
            net = slim.conv2d(net, out_channels, (1, 1), (1, 1),
                              activation_fn=None, padding='VALID')
        net = tf.pad(net, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
        net = deconv_layer(net, 4, out_channels, method=deconv)

    return net
