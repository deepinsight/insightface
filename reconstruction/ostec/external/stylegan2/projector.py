# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc

#----------------------------------------------------------------------------

class Projector:
    def __init__(self):
        self.num_steps                  = 1000
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.1
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5
        self.verbose                    = False
        self.clone_net                  = True

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._noise_in              = None
        self._dlatents_expr         = None
        self._images_expr           = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_network(self, Gs, minibatch_size=1):
        assert minibatch_size == 1
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Find dlatent stats.
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)[:, :1, :] # [N, 1, 512]
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True) # [1, 1, 512]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info('std = %g' % self._dlatent_std)

        # Find noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

        # Image output graph.
        self._info('Building image output graph...')
        self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var')
        self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
        dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._noise_in
        self._dlatents_expr = tf.tile(self._dlatents_var + dlatents_noise, [1, self._Gs.components.synthesis.input_shape[1], 1])
        self._images_expr = self._Gs.components.synthesis.get_output_for(self._dlatents_expr, randomize_noise=False)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        proc_images_expr = (self._images_expr + 1) * (255 / 2)
        sh = proc_images_expr.shape.as_list()
        if sh[2] > 256:
            factor = sh[2] // 256
            proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])

        # Loss graph.
        self._info('Building loss graph...')
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        if self._lpips is None:
            self._lpips = misc.load_pkl('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2') # vgg16_zhang_perceptual.pkl
        self._dist = self._lpips.get_output_for(proc_images_expr, self._target_images_var)
        self._loss = tf.reduce_sum(self._dist)

        # Noise regularization graph.
        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
                v = tf.reduce_mean(v, axis=[3, 5])
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self._dlatents_var] + self._noise_vars)
        self._opt_step = self._opt.apply_updates()

    def run(self, target_images):
        # Run to completion.
        self.start(target_images)
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results.
        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.noises = self.get_noises()
        pres.images = self.get_images()
        return pres

    def start(self, target_images):
        assert self._Gs is not None

        # Prepare target images.
        self._info('Preparing target images...')
        target_images = np.asarray(target_images, dtype='float32')
        target_images = (target_images + 1) * (255 / 2)
        sh = target_images.shape
        assert sh[0] == self._minibatch_size
        if sh[2] > self._target_images_var.shape[2]:
            factor = sh[2] // self._target_images_var.shape[2]
            target_images = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))

        # Initialize optimization state.
        self._info('Initializing optimization state...')
        tflib.set_vars({self._target_images_var: target_images, self._dlatents_var: np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])})
        tflib.run(self._noise_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')

        # Hyperparameters.
        t = self._cur_step / self.num_steps
        noise_strength = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Train.
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op)

        # Print status.
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            self._info('%-8d%-12g%-12g' % (self._cur_step, dist_value, loss_value))
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_expr, {self._noise_in: 0})

    def get_noises(self):
        return tflib.run(self._noise_vars)

    def get_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0})

#----------------------------------------------------------------------------
