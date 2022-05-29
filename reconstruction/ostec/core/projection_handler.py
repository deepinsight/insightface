# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import os
import argparse
import pickle
from tqdm.auto import tqdm
import PIL.Image
from PIL import ImageFilter
import numpy as np
import external.stylegan2.dnnlib.tflib as tflib
from external.stylegan2 import pretrained_networks
from core.generator_model import Generator
from core.perceptual_model import PerceptualModel, load_images
import external.stylegan2.dnnlib
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Projection_Handler():
    def __init__(self, args):
        self.args = args

        # Initialize generator and perceptual model
        tflib.init_tf()
        generator_network, discriminator_network, Gs_network = pretrained_networks.load_networks(args.model_url)
        self.generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise)

        if (args.dlatent_avg != ''):
            self.generator.set_dlatent_avg(np.load(args.dlatent_avg))

        perc_model = None
        if (args.use_lpips_loss > 0.00000001):
            if external.stylegan2.dnnlib.util.is_url(args.vgg_url):
                stream = external.stylegan2.dnnlib.util.open_url(args.vgg_url, cache_dir='../.stylegan2-cache')
            else:
                stream = open(args.vgg_url, 'rb')
            with stream as f:
                perc_model = pickle.load(f)


        self.perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=args.batch_size)

        self.ff_model = None
        if (self.ff_model is None):
            if os.path.exists(self.args.load_resnet):
                from keras.applications.resnet50 import preprocess_input
                print("Loading ResNet Model:")
                self.ff_model = load_model(self.args.load_resnet)
                # self.ff_model._make_predict_function()
                dummy_im = np.zeros([args.batch_size, args.resnet_image_size, args.resnet_image_size, 3], np.uint8)
                self.ff_model.predict(preprocess_input(dummy_im))
        if (self.ff_model is None):
            if os.path.exists(self.args.load_effnet):
                from efficientnet import preprocess_input
                print("Loading EfficientNet Model:")
                self.ff_model = load_model(self.args.load_effnet)

        self.perceptual_model.build_perceptual_model(self.generator, discriminator_network)
        self.perceptual_model.assign_placeholder('id_loss', args.use_id_loss)

    def run_projection(self, input_images, masks, heatmaps, id_features, iterations=None):
        n_iteration = self.args.iterations
        if iterations is not None:
            n_iteration = iterations
        return_imgs = {}
        return_dlatents = {}
        # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
        for names in  split_to_batches(list(input_images.keys()), self.args.batch_size):
            #split_to_batches(list(input_images.keys()), self.args.batch_size):
            #tqdm(split_to_batches(list(input_images.keys()), self.args.batch_size),
                                 #total=len(input_images) // self.args.batch_size):
            # tqdm._instances.clear()
            images_batch = [input_images[x] for x in names]
            masks_batch =  [masks[x] for x in names]
            heatmaps_batch =  [heatmaps[x] for x in names]
            # if args.output_video:
            #     video_out = {}
            #     for name in names:
            #         video_out[name] = cv2.VideoWriter(os.path.join(args.video_dir, f'{name}.avi'),
            #                                           cv2.VideoWriter_fourcc(*args.video_codec), args.video_frame_rate,
            #                                           (args.video_size, args.video_size))

            ## REGRESSION
            dlatents = None
            if (self.args.load_last != ''):  # load previous dlatents for initialization
                for name in names:
                    dl = np.expand_dims(np.load(os.path.join(self.args.load_last, f'{name}.npy')), axis=0)
                    if (dlatents is None):
                        dlatents = dl
                    else:
                        dlatents = np.vstack((dlatents, dl))
            else:
                if (self.ff_model is not None):  # predict initial dlatents with ResNet model
                    if (self.args.use_preprocess_input):
                        dlatents = self.ff_model.predict(
                            preprocess_input(load_images(images_batch, image_size=self.args.resnet_image_size)))
                    else:
                        dlatents = self.ff_model.predict(load_images(images_batch, image_size=self.args.resnet_image_size))
            if dlatents is not None:
                self.generator.set_dlatents(dlatents)

            ## OPTIMIZATION
            self.perceptual_model.set_reference_images(images_batch, masks_batch, heatmaps_batch, id_features)

            op = self.perceptual_model.optimize(self.generator.dlatent_variable, iterations=n_iteration)
            pbar = tqdm(op, leave=False, total=n_iteration)
            vid_count = 0
            best_loss = None
            best_dlatent = None
            avg_loss_count = 0
            if self.args.early_stopping:
                avg_loss = prev_loss = None

            for loss_dict in pbar:
                if self.args.early_stopping:  # early stopping feature
                    if prev_loss is not None:
                        if avg_loss is not None:
                            avg_loss = 0.5 * avg_loss + (prev_loss - loss_dict["loss"])
                            if avg_loss < self.args.early_stopping_threshold:  # count while under threshold; else reset
                                avg_loss_count += 1
                            else:
                                avg_loss_count = 0
                            if avg_loss_count > self.args.early_stopping_patience:  # stop once threshold is reached
                                print("")
                                break
                        else:
                            avg_loss = prev_loss - loss_dict["loss"]
                pbar.set_description(
                    " ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
                if best_loss is None or loss_dict["loss"] < best_loss:
                    if best_dlatent is None or self.args.average_best_loss <= 0.00000001:
                        best_dlatent = self.generator.get_dlatents()
                    else:
                        best_dlatent = 0.25 * best_dlatent + 0.75 * self.generator.get_dlatents()
                    if self.args.use_best_loss:
                        self.generator.set_dlatents(best_dlatent)
                    best_loss = loss_dict["loss"]
                # if self.args.output_video and (vid_count % self.args.video_skip == 0):
                #     batch_frames = self.generator.generate_images()
                #     for i, name in enumerate(names):
                #         video_frame = PIL.Image.fromarray(batch_frames[i], 'RGB').resize(
                #             (self.args.video_size, self.args.video_size), PIL.Image.LANCZOS)
                #         video_out[name].write(cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))
                self.generator.stochastic_clip_dlatents()
                prev_loss = loss_dict["loss"]
            if not self.args.use_best_loss:
                best_loss = prev_loss
            # pbar.set_postfix(loss="{:.4f}".format(best_loss))
            print(" ".join(names), " Loss {:.4f}".format(best_loss))

            # if self.args.output_video:
            #     for name in names:
            #         video_out[name].release()

            # Generate images from found dlatents and save them
            if self.args.use_best_loss:
                self.generator.set_dlatents(best_dlatent)

            generated_images = self.generator.generate_images()
            generated_dlatents = self.generator.get_dlatents()
            for img_array, dlatent, img_path, img_name in zip(generated_images, generated_dlatents, images_batch,
                                                              names):
                mask_img = None
                if self.args.composite_mask and (self.args.load_mask or self.args.face_mask):
                    _, im_name = os.path.split(img_path)
                    mask_img = os.path.join(self.args.mask_dir, f'{im_name}')
                if self.args.composite_mask and mask_img is not None and os.path.isfile(mask_img):
                    orig_img = PIL.Image.open(img_path).convert('RGB')
                    width, height = orig_img.size
                    imask = PIL.Image.open(mask_img).convert('L').resize((width, height))
                    imask = imask.filter(ImageFilter.GaussianBlur(self.args.composite_blur))
                    mask = np.array(imask) / 255
                    mask = np.expand_dims(mask, axis=-1)
                    img_array = mask * np.array(img_array) + (1.0 - mask) * np.array(orig_img)
                    img_array = img_array.astype(np.uint8)
                img = PIL.Image.fromarray(img_array, 'RGB')
                return_imgs[img_name] = img
                return_dlatents[img_name] = dlatent

            self.generator.reset_dlatents()
        return return_imgs, return_dlatents
