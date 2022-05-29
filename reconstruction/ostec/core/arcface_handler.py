# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import tensorflow as tf
import numpy as np
from external import arcface50
from skimage import transform as trans


def align_arcface(image, landmarks):
    """
    Aligns 'image' with its corresponding 'landmarks' to a predefined template
    with similarity transformation. This is the tensorflow implementation of
    the default alignment procedure of ArcFace
    Args:
        image: a 4D float32 numpy array with shape [batch_size, image_height,
        image_width, 3].
        landmarks: 68 iBug landmark points of 'image'. [batch_size, 68, 2]

    Returns:
        4-D float32 numpy array with shape [batch_size, 112, 112, 3]. Contains
        aligned version of 'image'

    """

    image_size = (112, 112)
    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    if image_size[1] == 112:
        dst[:, 0] += 8.0
    # dst = dst[:, ::-1]
    landmark5 = tf.stack([(landmarks[:, 36] + landmarks[:, 39]) / 2,
                          (landmarks[:, 42] + landmarks[:, 45]) / 2,
                          landmarks[:, 30],
                          landmarks[:, 48],
                          landmarks[:, 54]], 1)

    def py_similarity_transform(src, dst):
        tform = trans.SimilarityTransform()
        Ms = np.zeros([0, 3, 3], dtype=np.float32)
        for s in src:
            tform.estimate(s, dst)
            Ms = np.concatenate([Ms,[tform.params]],0)
        return Ms

    M = tf.py_func(py_similarity_transform, [landmark5, tf.constant(dst, 'float32')], tf.double, stateful=False)
    M.set_shape([image.get_shape().as_list()[0],3, 3])

    aligned = tf.contrib.image.transform(image, tf.cast(tf.contrib.image.matrices_to_flat_transforms(tf.map_fn(tf.linalg.inv,M)), 'float32'),interpolation='BILINEAR')

    return aligned[:, 0:image_size[0], 0:image_size[1], :]

def get_input_features(image, landmarks):
    """Extract features from a face recongnition networks including
        intermadiate activations. This function first align the image and
        then call identity_features()
    """
    image_aligned = align_arcface(image, landmarks)

    emb_norm, content, embedding, vars = identity_features(image_aligned,'id_features/')
    return emb_norm, vars, image_aligned #, content, embedding], image_aligned

def identity_features(input, name):
    """Extract features from a face recongnition networks including
        intermadiate activations.
    """
    with tf.variable_scope(name, reuse=True):
        input, embedding, content = arcface50.KitModel('models/fr_models/arcface50.npy', input * 255)
        emb_norm = tf.nn.l2_normalize(embedding, 1)
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return emb_norm, content, embedding, vars

class Arcface_Handler:
    def __init__(self):
        self.img_ph = tf.placeholder(dtype=tf.float32, name='img_ph', shape=[None, 256, 256, 3])
        self.lms_ph = tf.placeholder(dtype=tf.float32, name='lms_ph', shape=[None, 68, 2])
        aligned_img = align_arcface(self.img_ph, self.lms_ph)
        aligned_img.set_shape([None, 112, 112, 3])
        self.emb_norm, _, _, vars = identity_features(aligned_img, 'input_id_features')
        var_init = tf.variables_initializer(vars)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(var_init)

    def get_identity_features(self, img, lms):
        lms = (lms / img.shape[::-1]) * [256, 256]
        img = img.resize([256, 256])
        return self.sess.run(self.emb_norm, {self.img_ph: [img.pixels_with_channels_at_back()], self.lms_ph:[lms]})

def identity_features_numpy(image, landmarks, return_aligned=False):
    src_img = np.array([image])
    src_lms = np.array([landmarks])
    src_img = tf.constant(src_img,tf.float32)
    src_lms = tf.constant(src_lms,tf.float32)
    aligned_img = align_arcface(src_img,src_lms)
    emb_norm, _, _, vars = identity_features(aligned_img, 'input_id_features')
    var_init = tf.variables_initializer(vars)
    with tf.Session() as sess:
        sess.run(var_init)
        features = sess.run(emb_norm)
        if return_aligned:
            aligned = sess.run(aligned_img)

    tf.reset_default_graph()
    if return_aligned:
        return features, aligned
    else:
        return features

