# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Loss functions."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Logistic loss from the paper
# "Generative Adversarial Nets", Goodfellow et al. 2014

def G_logistic(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = -tf.nn.softplus(fake_scores_out) # log(1-sigmoid(fake_scores_out)) # pylint: disable=invalid-unary-operand-type
    return loss, None

def G_logistic_ns(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    return loss, None

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    return loss, None

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

def D_logistic_r2(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        fake_grads = tf.gradients(tf.reduce_sum(fake_scores_out), [fake_images_out])[0]
        gradient_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# WGAN loss from the paper
# "Wasserstein Generative Adversarial Networks", Arjovsky et al. 2017

def G_wgan(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = -fake_scores_out
    return loss, None

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, wgan_epsilon=0.001):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
        loss += epsilon_penalty * wgan_epsilon
    return loss, None

#----------------------------------------------------------------------------
# WGAN-GP loss from the paper
# "Improved Training of Wasserstein GANs", Gulrajani et al. 2017

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, wgan_lambda=10.0, wgan_epsilon=0.001, wgan_target=1.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = D.get_output_for(mixed_images_out, labels, is_training=True)
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_grads = tf.gradients(tf.reduce_sum(mixed_scores_out), [mixed_images_out])[0]
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
        reg = gradient_penalty * (wgan_lambda / (wgan_target**2))
    return loss, reg

#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

def G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------
