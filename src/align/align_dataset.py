"""Performs face alignment and stores face thumbnails in the output directory."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import random
import align_dlib  # @UnresolvedImport
import facenet

def main(args):
    align = align_dlib.AlignDlib(os.path.expanduser(args.dlib_face_predictor))
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    random.shuffle(dataset)
    # Scale the image such that the face fills the frame when cropped to crop_size
    scale = float(args.face_size) / args.image_size
    nrof_images_total = 0
    nrof_prealigned_images = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    if args.use_center_crop:
                        scaled = misc.imresize(img, args.prealigned_scale, interp='bilinear')
                        sz1 = scaled.shape[1]/2
                        sz2 = args.image_size/2
                        aligned = scaled[(sz1-sz2):(sz1+sz2),(sz1-sz2):(sz1+sz2),:]
                    else:
                        aligned = align.align(args.image_size, img, landmarkIndices=landmarkIndices, 
                                              skipMulti=False, scale=scale)
                    if aligned is not None:
                        print(image_path)
                        nrof_successfully_aligned += 1
                        misc.imsave(output_filename, aligned)
                    elif args.prealigned_dir:
                        # Face detection failed. Use center crop from pre-aligned dataset
                        class_name = os.path.split(output_class_dir)[1]
                        image_path_without_ext = os.path.join(os.path.expanduser(args.prealigned_dir), 
                                                              class_name, filename)
                        # Find the extension of the image
                        exts = ('jpg', 'png')
                        for ext in exts:
                            temp_path = image_path_without_ext + '.' + ext
                            image_path = ''
                            if os.path.exists(temp_path):
                                image_path = temp_path
                                break
                        try:
                            img = misc.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            scaled = misc.imresize(img, args.prealigned_scale, interp='bilinear')
                            sz1 = scaled.shape[1]/2
                            sz2 = args.image_size/2
                            cropped = scaled[(sz1-sz2):(sz1+sz2),(sz1-sz2):(sz1+sz2),:]
                            print(image_path)
                            nrof_prealigned_images += 1
                            misc.imsave(output_filename, cropped)
                    else:
                        print('Unable to align "%s"' % image_path)
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    print('Number of pre-aligned images: %d' % nrof_prealigned_images)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--dlib_face_predictor', type=str,
        help='File containing the dlib face predictor.', default='../data/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=110)
    parser.add_argument('--face_size', type=int,
        help='Size of the face thumbnail (height, width) in pixels.', default=96)
    parser.add_argument('--use_center_crop', 
        help='Use the center crop of the original image after scaling the image using prealigned_scale.', action='store_true')
    parser.add_argument('--prealigned_dir', type=str,
        help='Replace image with a pre-aligned version when face detection fails.', default='')
    parser.add_argument('--prealigned_scale', type=float,
        help='The amount of scaling to apply to prealigned images before taking the center crop.', default=0.87)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
