from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import menpo.io as mio
from menpo.image import Image
from menpo.shape import PointCloud
import cv2

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables

from menpo.transform import Translation, Scale
from menpo.shape import PointCloud

slim = tf.contrib.slim

def generate_heatmap(logits, num_classes):
    """Generates a coloured heatmap from the keypoint logits.

    Args:
        features: A `Tensor` of dimensions [num_batch, height, width, FLAGS.n_landmarks + 1].
    """

    keypoint_colours = np.array([plt.cm.spectral(x) for x in np.linspace(0, 1, num_classes + 1)])[
        ..., :3].astype(np.float32)

    prediction = tf.nn.softmax(logits)
    heatmap = tf.matmul(tf.reshape(prediction, (-1, num_classes + 1)), keypoint_colours)
    heatmap = tf.reshape(heatmap, (tf.shape(prediction)[0],
                                   tf.shape(prediction)[1],
                                   tf.shape(prediction)[2], 3))
    return heatmap

def generate_landmarks(keypoints):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    zeros = tf.to_float(tf.zeros_like(is_background))

    return tf.where(is_background, zeros, ones) * 255

def project_landmarks_to_shape_model(landmarks):
    final = []

    for lms in landmarks:
        lms = PointCloud(lms)
        similarity = AlignmentSimilarity(pca.global_transform.source, lms)
        projected_target = similarity.pseudoinverse().apply(lms)
        target = pca.model.reconstruct(projected_target)
        target = similarity.apply(target)
        final.append(target.points)

    return np.array(final).astype(np.float32)

def rescale_image(image, stride_width=64):
    # make sure smallest size is 600 pixels wide & dimensions are (k * stride_width) + 1
    height, width = image.shape

    # Taken from 'szross'
    scale_up = 625. / min(height, width)
    scale_cap = 961. / max(height, width)
    scale_up  = min(scale_up, scale_cap)
    new_height = stride_width * round((height * scale_up) / stride_width) + 1
    new_width = stride_width * round((width * scale_up) / stride_width) + 1
    image, tr = image.resize([new_height, new_width], return_transform=True)
    image.inverse_tr = tr
    return image

def frankotchellappa(dzdx, dzdy):
    from numpy.fft import ifftshift, fft2, ifft2
    rows, cols = dzdx.shape
    # The following sets up matrices specifying frequencies in the x and y
    # directions corresponding to the Fourier transforms of the gradient
    # data.  They range from -0.5 cycles/pixel to + 0.5 cycles/pixel.
    # The scaling of this is irrelevant as long as it represents a full
    # circle domain. This is functionally equivalent to any constant * pi
    pi_over_2 = np.pi / 2.0
    row_grid = np.linspace(-pi_over_2, pi_over_2, rows)
    col_grid = np.linspace(-pi_over_2, pi_over_2, cols)
    wy, wx = np.meshgrid(row_grid, col_grid, indexing='ij')

    # Quadrant shift to put zero frequency at the appropriate edge
    wx = ifftshift(wx)
    wy = ifftshift(wy)

    # Fourier transforms of gradients
    DZDX = fft2(dzdx)
    DZDY = fft2(dzdy)

    # Integrate in the frequency domain by phase shifting by pi/2 and
    # weighting the Fourier coefficients by their frequencies in x and y and
    # then dividing by the squared frequency
    denom = (wx ** 2 + wy ** 2)
    Z = (-1j * wx * DZDX - 1j * wy * DZDY) / denom
    Z = np.nan_to_num(Z)
    return np.real(ifft2(Z))

def line(image, x0, y0, x1, y1, color):
    steep = False
    if x0 < 0 or x0 >= 400 or x1 < 0 or x1 >= 400 or y0 < 0 or y0 >= 400 or y1 < 0 or y1 >= 400:
        return

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1) + 1):
        t = (x - x0) / float(x1 - x0)
        y = y0 * (1 - t) + y1 * t
        if steep:
            image[x, int(y)] = color
        else:
            image[int(y), x] = color

def draw_landmarks(img, lms):
    try:
        img = img.copy()

        for i, part in enumerate(parts_68[1:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]

                line(img, p2[1], p2[0], p1[1], p1[0], 1)
    except:
        pass
    return img

def batch_draw_landmarks(imgs, lms):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, lms)])

def build_graph(inputs, tree, transpose=(2,3,1,0), layers=[]):
    net = inputs

    if tree['name'] == 'nn.Sequential':
        with tf.name_scope('nn.Sequential'):
            for tr in tree['children']:
                net = build_graph(net, tr, transpose, layers)
    elif tree['name'] == 'nn.ConcatTable':
        net_table = []
        with tf.name_scope('nn.ConcatTable'):
            for tr in tree['children']:
                net_table.append(build_graph(net, tr, transpose, layers))
        net = net_table
    elif tree['name'] == 'nn.JoinTable':
        net = tf.concat(3, net)
    elif tree['name'] == 'nn.CAddTable':
        net = tf.add_n(net)
    elif tree['name'] == 'nn.SpatialConvolution':
        out_channel = int(tree['nOutputPlane'])
        kernal_shape = (int(tree['kH']),int(tree['kW']))
        stride_shape = (int(tree['dH']),int(tree['dW']))
        net = tf.pad(
                net, [
                    [0,0],
                    [int(tree['padH']),int(tree['padH'])],
                    [int(tree['padW']),int(tree['padW'])],
                    [0,0]
                ])
        if 'weight' in tree.keys() and 'bias' in tree.keys():
            net = slim.conv2d(net,
                              out_channel,
                              kernal_shape,
                              stride_shape,
                              activation_fn=None,
                              padding='VALID',
                              weights_initializer=tf.constant_initializer(tree['weight'].transpose(*transpose)),
                              biases_initializer=tf.constant_initializer(tree['bias'])
                             )
        else:
            net = slim.conv2d(net,
                              out_channel,
                              kernal_shape,
                              stride_shape,
                              activation_fn=None,
                              padding='VALID'
                             )

        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.SpatialFullConvolution':
        out_channel = int(tree['nOutputPlane'])
        kernal_shape = (int(tree['kH']),int(tree['kW']))
        stride_shape = (int(tree['dH']),int(tree['dW']))
        net = tf.pad(
                net, [
                    [0,0],
                    [int(tree['padH']),int(tree['padH'])],
                    [int(tree['padW']),int(tree['padW'])],
                    [0,0]
                ])
        if 'weight' in tree.keys() and 'bias' in tree.keys():
            net = slim.conv2d_transpose(net,
                              out_channel,
                              kernal_shape,
                              stride_shape,
                              activation_fn=None,
                              padding='VALID',
                              weights_initializer=tf.constant_initializer(tree['weight'].transpose(*transpose)),
                              biases_initializer=tf.constant_initializer(tree['bias'])
                             )
        else:
            net = slim.conv2d_transpose(net,
                              out_channel,
                              kernal_shape,
                              stride_shape,
                              activation_fn=None,
                              padding='VALID'
                             )
        tree['tfname'] = net.name
        tree['tfvar'] = net

    elif tree['name'] == 'nn.SpatialBatchNormalization':
        net = slim.nn.batch_normalization(net,
                                     tree['running_mean'],
                                     tree['running_var'],
                                     tree['bias'],
                                     tree['weight'],
                                     tree['eps'])
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.ReLU':
        net = slim.nn.relu(net)
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.Sigmoid':
        net = slim.nn.sigmoid(net)
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.SpatialMaxPooling':
        net = slim.max_pool2d(
            tf.pad(
                net, [
                    [0,0],
                    [int(tree['padH']),int(tree['padH'])],
                    [int(tree['padW']),int(tree['padW'])],
                    [0,0]
                ]),
            (int(tree['kH']),int(tree['kW'])),
            (int(tree['dH']),int(tree['dW']))
        )
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.Identity':
        pass
    else:
        raise Exception(tree['name'])

    return net

def build_graph_old(inputs, tree, transpose=(2,3,1,0)):
    net = inputs

    if tree['name'] == 'nn.Sequential':
        with tf.name_scope('nn.Sequential'):
            for tr in tree['children']:
                net = build_graph(net, tr, transpose, layers)
    elif tree['name'] == 'nn.ConcatTable':
        net_table = []
        with tf.name_scope('nn.ConcatTable'):
            for tr in tree['children']:
                net_table.append(build_graph(net, tr, transpose, layers))
        net = net_table
    elif tree['name'] == 'nn.JoinTable':
        net = tf.concat(3, net)
    elif tree['name'] == 'nn.CAddTable':
        net = tf.add_n(net)
    elif tree['name'] == 'nn.SpatialConvolution':
        out_channel = int(tree['nOutputPlane'])
        kernal_shape = (int(tree['kH']),int(tree['kW']))
        stride_shape = (int(tree['dH']),int(tree['dW']))
        net = tf.pad(
                net, [
                    [0,0],
                    [int(tree['padH']),int(tree['padH'])],
                    [int(tree['padW']),int(tree['padW'])],
                    [0,0]
                ])
        if 'weight' in tree.keys() and 'bias' in tree.keys():
            net = slim.conv2d(net,
                              out_channel,
                              kernal_shape,
                              stride_shape,
                              activation_fn=None,
                              padding='VALID',
                              weights_initializer=tf.constant_initializer(tree['weight'].transpose(*transpose)),
                              biases_initializer=tf.constant_initializer(tree['bias'])
                             )
        else:
            net = slim.conv2d(net,
                              out_channel,
                              kernal_shape,
                              stride_shape,
                              activation_fn=None,
                              padding='VALID'
                             )

        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.SpatialFullConvolution':
        out_channel = int(tree['nOutputPlane'])
        kernal_shape = (int(tree['kH']),int(tree['kW']))
        rate = np.min(int(tree['dH']),int(tree['dW']))
        h,w = tf.shape(net)[1:3]
        net = tf.image.resize_bilinear(net, (h,w,out_channel))

        tree['tfname'] = net.name
        tree['tfvar'] = net

    elif tree['name'] == 'nn.SpatialBatchNormalization':
        net = slim.batch_norm(net)
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.ReLU':
        net = slim.nn.relu(net)
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.Sigmoid':
        net = slim.nn.sigmoid(net)
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.SpatialMaxPooling':
        net = slim.max_pool2d(
            tf.pad(
                net, [
                    [0,0],
                    [int(tree['padH']),int(tree['padH'])],
                    [int(tree['padW']),int(tree['padW'])],
                    [0,0]
                ]),
            (int(tree['kH']),int(tree['kW'])),
            (int(tree['dH']),int(tree['dW']))
        )
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.Identity':
        pass
    else:
        raise Exception(tree['name'])

    return net

def keypts_encoding(keypoints, num_classes):
    keypoints = tf.to_int32(keypoints)
    keypoints = tf.reshape(keypoints, (-1,))
    keypoints = slim.layers.one_hot_encoding(keypoints, num_classes=num_classes+1)
    return keypoints

def get_weight(keypoints, mask=None, ng_w=0.01, ps_w=1.0):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    weights = tf.where(is_background, ones * ng_w, ones*ps_w)
    # if mask is not None:
    #     weights *= tf.to_float(mask)

    return weights

def ced_accuracy(t, dists):
    # Head	 Shoulder	Elbow	Wrist	Hip	   Knee	   Ankle
    pts_r  = tf.transpose(tf.gather(tf.transpose(dists), [8,12,11,10,2,1,0]))
    pts_l  = tf.transpose(tf.gather(tf.transpose(dists), [9,13,14,15,3,4,5]))
    part_pckh = (tf.to_int32(pts_r <= t) + tf.to_int32(pts_l <= t)) / 2

    return tf.concat(1, [part_pckh, tf.reduce_sum(tf.to_int32(dists <= t), 1)[...,None] / tf.shape(dists)[1]])

def pckh(preds, gts, scales):
    t_range = np.arange(0,0.51,0.01)
    dists = tf.sqrt(tf.reduce_sum(tf.pow(preds - gts, 2), reduction_indices=-1)) / scales
    # pckh = [ced_accuracy(t, dists) for t in t_range]
    # return pckh[-1]
    return ced_accuracy(0.5, dists)

def import_image(img_path):
    img = cv2.imread(str(img_path))
    original_image = Image.init_from_channels_at_back(img[:,:,-1::-1])

    try:
        original_image_lms = mio.import_landmark_file('{}/{}.ljson'.format(img_path.parent, img_path.stem)).lms.points.astype(np.float32)
        original_image.landmarks['LJSON'] = PointCloud(original_image_lms)
    except:
        pass

    return original_image

def crop_image(img, center, scale, res, base=384):
    h = base * scale

    t = Translation(
        [
            res[0] * (-center[0] / h + .5),
            res[1] * (-center[1] / h + .5)
        ]).compose_after(Scale((res[0] / h, res[1] / h))).pseudoinverse()


    # Upper left point
    ul = np.floor(t.apply([0,0]))
    # Bottom right point
    br = np.ceil(t.apply(res).astype(np.int))

    # crop and rescale

    cimg, trans = img.warp_to_shape(br-ul, Translation(-(br-ul)/2+(br+ul)/2) ,return_transform=True)
    c_scale = np.min(cimg.shape) / np.mean(res)
    new_img = cimg.rescale(1 / c_scale).resize(res)
    return new_img, trans, c_scale
