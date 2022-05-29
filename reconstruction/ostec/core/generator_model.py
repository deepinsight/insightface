# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import math
import tensorflow as tf
import numpy as np
import external.stylegan2.dnnlib.tflib as tflib
from functools import partial


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size, tiled_dlatent, model_scale=18, tile_size = 1):
    if tiled_dlatent:
        low_dim_dlatent = tf.get_variable('learnable_dlatents',
            shape=(batch_size, tile_size, 512),
            dtype='float32',
            initializer=tf.initializers.random_normal())
        return tf.tile(low_dim_dlatent, [1, model_scale // tile_size, 1])
    else:
        return tf.get_variable('learnable_dlatents',
            shape=(batch_size, model_scale, 512),
            dtype='float32',
            initializer=tf.initializers.random_normal())


class Generator:
    def __init__(self, model, batch_size, custom_input=None, clipping_threshold=2, tiled_dlatent=False, model_res=1024, randomize_noise=False):
        self.batch_size = batch_size
        self.tiled_dlatent=tiled_dlatent
        self.model_scale = int(2*(math.log(model_res,2)-1)) # For example, 1024 -> 18

        if tiled_dlatent:
            self.initial_dlatents = np.zeros((self.batch_size, 512))
            model.components.synthesis.run(np.zeros((self.batch_size, self.model_scale, 512)),
                randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, tiled_dlatent=True),
                                                partial(create_stub, batch_size=batch_size)],
                structure='fixed')
        else:
            self.initial_dlatents = np.zeros((self.batch_size, self.model_scale, 512))
            if custom_input is not None:
                model.components.synthesis.run(self.initial_dlatents,
                    randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                    custom_inputs=[partial(custom_input.eval(), batch_size=batch_size), partial(create_stub, batch_size=batch_size)],
                    structure='fixed')
            else:
                model.components.synthesis.run(self.initial_dlatents,
                    randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                    custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, tiled_dlatent=False, model_scale=self.model_scale),
                                                    partial(create_stub, batch_size=batch_size)],
                    structure='fixed')

        self.dlatent_avg_def = model.get_var('dlatent_avg')
        self.reset_dlatent_avg()
        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
        self._assign_dlatent_ph = tf.placeholder(tf.float32, name="assign_dlatent_ph")
        self._assign_dlantent = tf.assign(self.dlatent_variable, self._assign_dlatent_ph)
        self.set_dlatents(self.initial_dlatents)

        def get_tensor(name):
            try:
                return self.graph.get_tensor_by_name(name)
            except KeyError:
                return None

        self.generator_output = get_tensor('G_synthesis_1/_Run/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis_1/_Run/concat/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis_1/_Run/concat_1/concat:0')
        # If we loaded only Gs and didn't load G or D, then scope "G_synthesis_1" won't exist in the graph.
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis/_Run/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis/_Run/concat/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis/_Run/concat_1/concat:0')
        if self.generator_output is None:
            for op in self.graph.get_operations():
                print(op)
            raise Exception("Couldn't find G_synthesis_1/_Run/concat tensor output")
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

        # Implement stochastic clipping similar to what is described in https://arxiv.org/abs/1702.04782
        # (Slightly different in that the latent space is normal gaussian here and was uniform in [-1, 1] in that paper,
        # so we clip any vector components outside of [-2, 2]. It seems fine, but I haven't done an ablation check.)
        clipping_mask = tf.math.logical_or(self.dlatent_variable > clipping_threshold, self.dlatent_variable < -clipping_threshold)
        clipped_values = tf.where(clipping_mask, tf.random_normal(shape=self.dlatent_variable.shape), self.dlatent_variable)
        self.stochastic_clip_op = tf.assign(self.dlatent_variable, clipped_values)

    def reset_dlatents(self):
        self.set_dlatents(self.initial_dlatents)

    def set_dlatents(self, dlatents):
        if self.tiled_dlatent:
            if (dlatents.shape != (self.batch_size, 512)) and (dlatents.shape[1] != 512):
                dlatents = np.mean(dlatents, axis=1)
            if (dlatents.shape != (self.batch_size, 512)):
                dlatents = np.vstack([dlatents, np.zeros((self.batch_size-dlatents.shape[0], 512))])
            assert (dlatents.shape == (self.batch_size, 512))
        else:
            if (dlatents.shape[1] > self.model_scale):
                dlatents = dlatents[:,:self.model_scale,:]
            if (isinstance(dlatents.shape[0], int)):
                if (dlatents.shape != (self.batch_size, self.model_scale, 512)):
                    dlatents = np.vstack([dlatents, np.zeros((self.batch_size-dlatents.shape[0], self.model_scale, 512))])
                assert (dlatents.shape == (self.batch_size, self.model_scale, 512))
                self.sess.run([self._assign_dlantent], {self._assign_dlatent_ph: dlatents})
                return
            else:
                self._assign_dlantent = tf.assign(self.dlatent_variable, dlatents)
                return
        self.sess.run([self._assign_dlantent], {self._assign_dlatent_ph: dlatents})

    def stochastic_clip_dlatents(self):
        self.sess.run(self.stochastic_clip_op)

    def get_dlatents(self):
        return self.sess.run(self.dlatent_variable)

    def get_dlatent_avg(self):
        return self.dlatent_avg

    def set_dlatent_avg(self, dlatent_avg):
        self.dlatent_avg = dlatent_avg

    def reset_dlatent_avg(self):
        self.dlatent_avg = self.dlatent_avg_def

    def generate_images(self, dlatents=None):
        if dlatents is not None:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)