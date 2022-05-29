# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Inception Score (IS)."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class IS(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('https://drive.google.com/uc?id=1Mz9zQnIrusm3duZB91ng_aUIePFNI6Jx') # inception_v3_softmax.pkl
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **Gs_kwargs)
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(inception_clone.get_output_for(images))

        # Calculate activations for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]

        # Calculate IS.
        scores = []
        for i in range(self.num_splits):
            part = activations[i * self.num_images // self.num_splits : (i + 1) * self.num_images // self.num_splits]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        self._report_result(np.mean(scores), suffix='_mean')
        self._report_result(np.std(scores), suffix='_std')

#----------------------------------------------------------------------------
