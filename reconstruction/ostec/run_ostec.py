# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import time
import os
import glob
from random import shuffle
import argparse
from argparse import Namespace
import menpo.io as mio
import menpo.image
import cv2
import sys
sys.path.append("external/stylegan2")
sys.path.append("external/deep3dfacerecon")
sys.path.append("external/graphonomy")
from core.operator import Operator
from core.config import get_config
import numpy as np
from utils.utils import im_menpo2PIL, fix_obj
from external.face_detector.detect_face import Face_Detector
from FaceHairMask.MaskExtractor import MaskExtractor
from menpo.shape import TexturedTriMesh
import menpo3d.io as m3io

def main(args):
    source_dir = args.source_dir
    save_dir = args.save_dir
    os.makedirs(save_dir,exist_ok=True)
    operator = Operator(args)
    detector = Face_Detector()
    if not args.ganfit:
        from external.deep3dfacerecon.ostec_api import Deep3dModel
        deep3dmodel = Deep3dModel()
    maskExtractor = MaskExtractor()


    # while True:
    for ext in ['.png', '.jpg']:
        print('Scanning paths...')
        paths = glob.glob(source_dir + '/*' + ext)
        shuffle(paths)
        for path in paths:
            # try: # To avoid detection errors on large datasets
            save_path = path.replace(source_dir, save_dir)
            pkl_path = path.replace(ext,'.pkl')
            if not os.path.isfile(save_path.replace(ext, '.png')):
                print('Started: ' + path)
                start = time.time()

                img = menpo.image.Image(np.transpose(cv2.imread(path)[:,:,::-1],[2,0,1])/255.0)

                if args.ganfit and not os.path.isfile(pkl_path):
                    raise Exception('Reconstruction from GANfit mode is activated and no GANFit reconstruction pickle file has been found! Either Remove --ganfit flag or Run GANFit first.')

                if os.path.isfile(pkl_path): # GANFit mode
                    fitting = mio.import_pickle(pkl_path)

                else: # Deep3dReconstruction mode
                    _, lms = detector.face_detection((img.pixels_with_channels_at_back() * 255).astype(np.uint8))
                    fitting = deep3dmodel.recontruct(im_menpo2PIL(img), lms)
                    img = menpo.image.Image(fitting['input'])

                _, face_mask = maskExtractor.main(img)

                final_uv, results_dict = operator.run(img, fitting, face_mask)
                tmesh = TexturedTriMesh(fitting['vertices'],operator.tcoords.points,final_uv,operator.uv_trilist)
                m3io.export_textured_mesh(tmesh,save_path.replace(ext, '.obj'),texture_extension='.png')
                fix_obj(save_path.replace(ext, '.obj'))
                # mio.export_image(final_uv, save_path.replace(ext, '.png'))

                if args.frontalize:
                    mio.export_image(results_dict['frontal'], save_path.replace(ext,'_frontal.png'))
                if args.pickle:
                    mio.export_pickle(results_dict, save_path.replace(ext,'.pkl'))

                print('Total Processing Time : %.2f secs' % (time.time() - start))

            # except Exception as inst:
            #     print(type(inst))  # the exception instance
            #     print(inst.args)  # arguments stored in .args
            #     print(inst)  # __str__ allows args to be printed directly,

if __name__ == "__main__":
    args, unparsed = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', help='Directory of input 2D images')
    parser.add_argument('--save_dir', help='Directory to save synthesized UVs')
    args2 = parser.parse_args(unparsed)
    args = Namespace(**vars(args), **vars(args2))
    main(args)
