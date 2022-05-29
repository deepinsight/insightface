# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import bz2
import PIL.Image
from PIL import ImageFilter
import numpy as np
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K
import traceback
import external.stylegan2.dnnlib.tflib as tflib
from core.landmark_handler import Landmark_Handler
from core import arcface_handler


def load_images(images_PIL, image_size=256, sharpen=False, im_type='RGB'):
    loaded_images = list()
    for img_pil in images_PIL:
      img = img_pil.convert(im_type)
      if image_size is not None:
        img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
        if (sharpen):
            img = img.filter(ImageFilter.DETAIL)
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

def tf_custom_adaptive_loss(a,b):
    from adaptive import lossfun
    shape = a.get_shape().as_list()
    dim = np.prod(shape[1:])
    a = tf.reshape(a, [-1, dim])
    b = tf.reshape(b, [-1, dim])
    loss, _, _ = lossfun(b-a, var_suffix='1')
    return tf.math.reduce_mean(loss)

def tf_custom_adaptive_rgb_loss(a,b):
    from adaptive import image_lossfun
    loss, _, _ = image_lossfun(b-a, color_space='RGB', representation='PIXEL')
    return tf.math.reduce_mean(loss)

def tf_custom_l1_loss(img1,img2):
  return tf.math.reduce_mean(tf.math.abs(img2-img1), axis=None)

def tf_custom_logcosh_loss(img1,img2):
  return tf.math.reduce_mean(tf.keras.losses.logcosh(img1,img2))

def create_stub(batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class PerceptualModel:
    def __init__(self, args, batch_size=1, perc_model=None, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.args = args
        self.lr = args.lr
        self.decay_rate = args.decay_rate
        self.decay_steps = args.decay_steps
        self.img_size = args.image_size
        self.layer = args.use_vgg_layer
        self.vgg_loss = args.use_vgg_loss
        self.face_mask = args.face_mask
        if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
            self.vgg_loss = None
        self.pixel_loss = args.use_pixel_loss
        if (self.pixel_loss <= self.epsilon):
            self.pixel_loss = None
        self.mssim_loss = args.use_mssim_loss
        if (self.mssim_loss <= self.epsilon):
            self.mssim_loss = None
        self.lpips_loss = args.use_lpips_loss
        if (self.lpips_loss <= self.epsilon):
            self.lpips_loss = None
        self.l1_penalty = args.use_l1_penalty
        if (self.l1_penalty <= self.epsilon):
            self.l1_penalty = None
        self.adaptive_loss = args.use_adaptive_loss
        self.sharpen_input = args.sharpen_input
        self.batch_size = batch_size
        if perc_model is not None and self.lpips_loss is not None:
            self.perc_model = perc_model
        else:
            self.perc_model = None
        self.ref_img = None
        self.ref_weight = None
        self.ref_heatmaps = None
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None
        self.discriminator_loss = args.use_discriminator_loss
        if (self.discriminator_loss <= self.epsilon):
            self.discriminator_loss = None
        if self.discriminator_loss is not None:
            self.discriminator = None
            self.stub = create_stub(batch_size)
        self.landmark_loss = args.use_landmark_loss



    def add_placeholder(self, var_name):
        var_val = getattr(self, var_name)
        setattr(self, var_name + "_placeholder", tf.placeholder(var_val.dtype, shape=var_val.get_shape()))
        setattr(self, var_name + "_op", var_val.assign(getattr(self, var_name + "_placeholder")))

    def assign_placeholder(self, var_name, var_val):
        self.sess.run(getattr(self, var_name + "_op"), {getattr(self, var_name + "_placeholder"): var_val})

    def build_perceptual_model(self, generator, discriminator=None):
        # Learning rate
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        incremented_global_step = tf.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        self.learning_rate = tf.train.exponential_decay(self.lr, incremented_global_step,
                self.decay_steps, self.decay_rate, staircase=True)
        self.sess.run([self._reset_global_step])

        if self.discriminator_loss is not None:
            self.discriminator = discriminator

        generated_image_tensor = generator.generated_image
        generated_image = tf.image.resize_nearest_neighbor(generated_image_tensor,
                                                                  (self.img_size, self.img_size), align_corners=True)
        self.generated_img = generated_image
        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.ref_weight = tf.get_variable('ref_weight', shape=generated_image.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder("ref_img")
        self.add_placeholder("ref_weight")

        if (self.vgg_loss is not None):
            vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
            self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
            generated_img_features = self.perceptual_model(preprocess_input(self.ref_weight * generated_image))
            dummy_im = np.zeros([self.args.batch_size, self.img_size, self.img_size, 3],np.float32)
            self.perceptual_model.predict_on_batch(dummy_im)
            self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
            self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
            self.sess.run([self.features_weight.initializer, self.features_weight.initializer])
            self.add_placeholder("ref_img_features")
            self.add_placeholder("features_weight")

        landmark_model = Landmark_Handler(self.args, self.sess, generated_image/255)
        landmark_model.load_model()
        ibug84to68_ind = list(range(0, 33, 2)) + list(range(33, 84))
        self.generated_heatmaps = tf.gather(landmark_model.lms_heatmap_prediction, ibug84to68_ind, axis=3)
        self.generated_landmarks = tf.gather(landmark_model.pts_predictions, ibug84to68_ind, axis=1)

        self.ref_heatmaps = tf.get_variable('ref_heatmaps', shape=self.generated_heatmaps.shape,
                                            dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder("ref_heatmaps")

        self.generated_id, vars, _ = arcface_handler.get_input_features(generated_image / 255, self.generated_landmarks[:, :, ::-1])
        self.init_id_vars = tf.variables_initializer(vars)

        self.org_features = tf.get_variable('org_features', shape=self.generated_id.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder("org_features")
        self.id_loss = tf.get_variable('id_loss', shape=(),
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder('id_loss')

        if self.perc_model is not None and self.lpips_loss is not None:
            img1 = tflib.convert_images_from_uint8(self.ref_weight * self.ref_img, nhwc_to_nchw=True)
            img2 = tflib.convert_images_from_uint8(self.ref_weight * generated_image, nhwc_to_nchw=True)

        self.loss = 0
        # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            if self.adaptive_loss:
                self.loss += self.vgg_loss * tf_custom_adaptive_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
            else:
                self.loss += self.vgg_loss * tf_custom_logcosh_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
        # + logcosh loss on image pixels
        if (self.pixel_loss is not None):
            if self.adaptive_loss:
                self.loss += self.pixel_loss * tf_custom_adaptive_rgb_loss(self.ref_weight * self.ref_img, self.ref_weight * generated_image)
            else:
                self.loss += self.pixel_loss * tf_custom_logcosh_loss(self.ref_weight * self.ref_img, self.ref_weight * generated_image)
        # + MS-SIM loss on image pixels
        if (self.mssim_loss is not None):
            self.loss += self.mssim_loss * tf.math.reduce_mean(1-tf.image.ssim_multiscale(self.ref_weight * self.ref_img, self.ref_weight * generated_image, 1))
        # + extra perceptual loss on image pixels
        if self.perc_model is not None and self.lpips_loss is not None:
            self.loss += self.lpips_loss * tf.math.reduce_mean(self.perc_model.get_output_for(img1, img2))
        # + L1 penalty on dlatent weights
        if self.l1_penalty is not None:
            self.loss += self.l1_penalty * 512 * tf.math.reduce_mean(tf.math.abs(generator.dlatent_variable-generator.get_dlatent_avg()))
        # discriminator loss (realism)
        if self.discriminator_loss is not None:
            self.loss += self.discriminator_loss * tf.math.reduce_mean(self.discriminator.get_output_for(
                tflib.convert_images_from_uint8(generated_image_tensor, nhwc_to_nchw=True), self.stub))
        # - discriminator_network.get_output_for(tflib.convert_images_from_uint8(ref_img, nhwc_to_nchw=True), stub)
        if self.landmark_loss is not None:
            self.loss += self.landmark_loss * tf.math.reduce_mean(tf.reduce_sum(tf.pow(self.ref_heatmaps - self.generated_heatmaps, 2), 2))
        if self.id_loss is not None:
            self.id_loss_comp = tf.losses.cosine_distance(self.generated_id, self.org_features, 1)
            self.loss += self.id_loss * self.id_loss_comp

        # Define Optimizer
        vars_to_optimize = generator.dlatent_variable
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        if self.args.optimizer == 'lbfgs':
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list=vars_to_optimize, method='L-BFGS-B', options={'maxiter': self.args.iterations})
        else:
            if self.args.optimizer == 'ggt':
                self.optimizer = tf.contrib.opt.GGTOptimizer(learning_rate=self.learning_rate)
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.min_op = self.optimizer.minimize(self.loss, var_list=[vars_to_optimize])
            self.sess.run(tf.variables_initializer(self.optimizer.variables()))
        #min_op = optimizer.minimize(self.sess)
        #optim_results = tfp.optimizer.lbfgs_minimize(make_val_and_grad_fn(get_loss), initial_position=vars_to_optimize, num_correction_pairs=10, tolerance=1e-8)
        K.manual_variable_initialization(True)
        self.sess.graph.finalize()  # Graph is read-only after this statement.


    def set_reference_images(self, images_PIL, masks_PIL, heatmaps, id_features):
        assert(len(images_PIL) != 0 and len(images_PIL) <= self.batch_size)
        loaded_image = load_images(images_PIL, self.img_size, sharpen=self.sharpen_input)
        loaded_mask = load_images(masks_PIL, self.img_size, sharpen=self.sharpen_input, im_type='L')
        heatmaps = np.transpose(np.array(heatmaps), [0, 2, 3, 1])
        input_size = np.array(heatmaps).shape[2]
        output_size = int(self.ref_heatmaps.shape[1])
        bin_size = input_size // output_size
        loaded_heatmaps = heatmaps.reshape((heatmaps.shape[0], output_size, bin_size,
                                             output_size, bin_size, 68)).max(4).max(2)
        image_features = None
        if self.perceptual_model is not None:
            image_features = self.perceptual_model.predict_on_batch(preprocess_input(np.array(loaded_image)))
            weight_mask = np.ones(self.features_weight.shape)

        if self.face_mask:
            image_mask = np.zeros(self.ref_weight.shape)
            for (i, (im, mask)) in enumerate(zip(loaded_image, loaded_mask)):
                try:
                    mask = np.array(mask)/255
                    mask = np.expand_dims(mask,axis=-1)
                    mask = np.ones(im.shape,np.float32) * mask #?
                except Exception as e:
                    print("Exception in mask handling for mask")
                    traceback.print_exc()
                    mask = np.ones(im.shape[:2],np.uint8)
                    mask = np.ones(im.shape,np.float32) * np.expand_dims(mask,axis=-1)
                image_mask[i] = mask
            img = None
        else:
            image_mask = np.ones(self.ref_weight.shape)

        if len(images_PIL) != self.batch_size:
            if image_features is not None:
                features_space = list(self.features_weight.shape[1:])
                existing_features_shape = [len(images_PIL)] + features_space
                empty_features_shape = [self.batch_size - len(images_PIL)] + features_space
                existing_examples = np.ones(shape=existing_features_shape)
                empty_examples = np.zeros(shape=empty_features_shape)
                weight_mask = np.vstack([existing_examples, empty_examples])
                image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

            images_space = list(self.ref_weight.shape[1:])
            existing_images_space = [len(images_PIL)] + images_space
            empty_images_space = [self.batch_size - len(images_PIL)] + images_space
            existing_images = np.ones(shape=existing_images_space)
            empty_images = np.zeros(shape=empty_images_space)
            image_mask = image_mask * np.vstack([existing_images, empty_images])
            loaded_image = np.vstack([loaded_image, np.zeros(empty_images_space)])

        if image_features is not None:
            self.assign_placeholder("features_weight", weight_mask)
            self.assign_placeholder("ref_img_features", image_features)
        self.assign_placeholder("ref_weight", image_mask)
        self.assign_placeholder("ref_img", loaded_image)
        self.assign_placeholder("org_features", id_features)
        self.assign_placeholder("ref_heatmaps", loaded_heatmaps)

    def optimize(self, vars_to_optimize, iterations=200):
        self.sess.run(self._reset_global_step)
        self.sess.run(self.init_id_vars)
        fetch_ops = [self.min_op, self.loss, self.id_loss_comp, self.learning_rate]
        for _ in range(iterations):
            if self.args.optimizer == 'lbfgs':
                self.optimizer.minimize(self.sess, fetches=[vars_to_optimize, self.loss])
                yield {"loss":self.loss.eval()}
            else:
                _, loss, id_loss, lr = self.sess.run(fetch_ops)
                yield {"loss":loss,"id_loss":id_loss,"lr":lr}
