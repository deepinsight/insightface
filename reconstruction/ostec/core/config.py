# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import argparse

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

parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')

parser.add_argument('-m', '--mode',default='hard', choices=['soft', 'auto', 'hard'],
                    help='Soft: keep original texture, Hard: generate all, auto: soft for frontal, hard for profile')
parser.add_argument('-f', '--frontalize', action='store_true', help='Run frontalization at the end')
parser.add_argument('-p', '--pickle', action='store_true', help='Save pickle with everything')
parser.add_argument('-g', '--ganfit', action='store_true', help='Reconstruction from GANFit is a must. If not raised, it is automatic: GANFit rec. if pickle found, Deep3DRecon otherwise.')
parser.add_argument('--iterations_frontalize', default=300, help='Number of optimization steps for each batch', type=int)

parser.add_argument('--load_last', default='', help='Start with embeddings from directory')
parser.add_argument('--dlatent_avg', default='', help='Use dlatent from file specified here for truncation instead of dlatent_avg from Gs')
parser.add_argument('--model_url', default='gdrive:networks/stylegan2-ffhq-config-f.pkl', help='Fetch a StyleGAN model to train on from this URL') # default='gdrive:networks/stylegan2-ffhq-config-f.pkl'
parser.add_argument('--model_res', default=1024, help='The dimension of images in the StyleGAN model', type=int)
parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
parser.add_argument('--optimizer', default='ggt', help='Optimization algorithm used for optimizing dlatents')

# Perceptual model params
parser.add_argument('--vgg_url', default='models/vgg16_zhang_perceptual.pkl', help='Fetch VGG model on from this URL') # default='https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2'
parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
parser.add_argument('--resnet_image_size', default=224, help='Size of images for the Resnet model', type=int)
parser.add_argument('--lr', default=0.25, help='Learning rate for perceptual model', type=float)
parser.add_argument('--decay_rate', default=0.9, help='Decay rate for learning rate', type=float)
parser.add_argument('-i', '--iterations', default=200, help='Number of optimization steps for each batch', type=int)
parser.add_argument('--decay_steps', default=4, help='Decay steps for learning rate decay (as a percent of iterations)', type=float)
parser.add_argument('--early_stopping', default=True, help='Stop early once training stabilizes', type=str2bool, nargs='?', const=True)
parser.add_argument('--early_stopping_threshold', default=0.5, help='Stop after this threshold has been reached', type=float)
parser.add_argument('--early_stopping_patience', default=10, help='Number of iterations to wait below threshold', type=int)
parser.add_argument('--load_effnet', default='data/finetuned_effnet.h5', help='Model to load for EfficientNet approximation of dlatents')
parser.add_argument('--load_resnet', default='models/resnet_18_20191231.h5', help='Model to load for ResNet approximation of dlatents')
parser.add_argument('--use_preprocess_input', default=True, help='Call process_input() first before using feed forward net', type=str2bool, nargs='?', const=True)
parser.add_argument('--use_best_loss', default=True, help='Output the lowest loss value found as the solution', type=str2bool, nargs='?', const=True)
parser.add_argument('--average_best_loss', default=0.25, help='Do a running weighted average with the previous best dlatents found', type=float)
parser.add_argument('--sharpen_input', default=True, help='Sharpen the input images', type=str2bool, nargs='?', const=True)
parser.add_argument('--landmark_model', default='./models/alignment/3D84/model.ckpt-277538', help='Landmark model path')

# Loss function options
parser.add_argument('--use_vgg_loss', default=0.4, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_vgg_layer', default=9, help='Pick which VGG layer to use.', type=int)
parser.add_argument('--use_pixel_loss', default=1.5, help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_mssim_loss', default=200, help='Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_lpips_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_l1_penalty', default=0.5, help='Use L1 penalty on latents; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_discriminator_loss', default=0.5, help='Use trained discriminator to evaluate realism.', type=float)
parser.add_argument('--use_adaptive_loss', default=False, help='Use the adaptive robust loss function from Google Research for pixel and VGG feature loss.', type=str2bool, nargs='?', const=True)
parser.add_argument('--use_landmark_loss', default=200, help='Use landmark loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_id_loss', default=10, help='Use landmark loss; 0 to disable, > 0 to scale.', type=float)
parser.add_argument('--use_id_loss_frontalize', default=100, help='Use landmark loss; 0 to disable, > 0 to scale.', type=float)

# Generator params
parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=str2bool, nargs='?', const=True)
parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=str2bool, nargs='?', const=True)
parser.add_argument('--clipping_threshold', default=2.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)

# Masking params
parser.add_argument('--load_mask', default=False, help='Load segmentation masks', type=str2bool, nargs='?', const=True)
parser.add_argument('--face_mask', default=True, help='Generate a mask for predicting only the face area', type=str2bool, nargs='?', const=True)
parser.add_argument('--use_grabcut', default=True, help='Use grabcut algorithm on the face mask to better segment the foreground', type=str2bool, nargs='?', const=True)
parser.add_argument('--scale_mask', default=1.4, help='Look over a wider section of foreground for grabcut', type=float)
parser.add_argument('--composite_mask', default=False, help='Merge the unmasked area back into the generated image', type=str2bool, nargs='?', const=True)
parser.add_argument('--composite_blur', default=8, help='Size of blur filter to smoothly composite the images', type=int)

# Video params
parser.add_argument('--video_dir', default='videos', help='Directory for storing training videos')
parser.add_argument('--output_video', default=False, help='Generate videos of the optimization process', type=bool)
parser.add_argument('--video_codec', default='MJPG', help='FOURCC-supported video codec name')
parser.add_argument('--video_frame_rate', default=24, help='Video frames per second', type=int)
parser.add_argument('--video_size', default=512, help='Video size in pixels', type=int)
parser.add_argument('--video_skip', default=1, help='Only write every n frames (1 = write every frame)', type=int)


def get_config():
    args, other_args = parser.parse_known_args()
    args.decay_steps *= 0.01 * args.iterations  # Calculate steps as a percent of total iterations

    return args, other_args