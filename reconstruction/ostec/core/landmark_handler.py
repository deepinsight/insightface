# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import tensorflow as tf
from external.landmark_detector import networks
from external.landmark_detector.flags import FLAGS


def tf_heatmap_to_lms(heatmap):
    hs = tf.argmax(tf.reduce_max(heatmap, 2), 1)
    ws = tf.argmax(tf.reduce_max(heatmap, 1), 1)
    lms = tf.transpose(tf.to_float(tf.stack([hs, ws])), perm=[1, 2, 0])
    return lms

class Landmark_Handler():
    def __init__(self, args, sess, generated_image):
        self.sess = sess

        self.model_path = args.landmark_model
        n_landmarks = 84
        FLAGS.n_landmarks = 84

        net_model = networks.DNFaceMultiView('')
        with tf.variable_scope('net'):
            self.lms_heatmap_prediction, states = net_model._build_network(generated_image, datas=None, is_training=False,
                                                                      n_channels=n_landmarks)
            self.pts_predictions = tf_heatmap_to_lms(self.lms_heatmap_prediction)
            variables = tf.all_variables()
            variables_to_restore = [v for v in variables if v.name.split('/')[0] == 'net']
            self.saver = tf.train.Saver(variables_to_restore)

    def load_model(self):
        self.saver.restore(self.sess, self.model_path)
