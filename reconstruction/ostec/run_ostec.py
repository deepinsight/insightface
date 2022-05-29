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
import sys
sys.path.append("external/stylegan2")
sys.path.append("external/deep3dfacerecon")
from core.operator import Operator
from core.config import get_config
import numpy as np
from utils.utils import im_menpo2PIL
from external.deep3dfacerecon.ostec_api import Deep3dModel
from external.face_detector.detect_face import Face_Detector

def main(args):
    source_dir = args.source_dir
    save_dir = args.save_dir
    operator = Operator(args)
    detector = Face_Detector()
    deep3dmodel = Deep3dModel()

    # while True:
    for ext in ['.png', '.jpg']:
        print('Scanning paths...')
        paths = glob.glob(source_dir + '/*' + ext)
        shuffle(paths)
        for path in paths:
            # try: # To avoid detection errors on large datasets
            save_path = path.replace(source_dir, save_dir)
            pkl_path = path.replace(ext,'.pkl')
            if not os.path.isfile(save_path.replace(ext, '_uv.'+ext)):
                print('Started: ' + path)
                start = time.time()
                img = mio.import_image(path)

                if os.path.isfile(pkl_path): # GANFit mode
                    fitting = mio.import_pickle(pkl_path)

                else: # Deep3dReconstruction mode
                    _, lms = detector.face_detection((img.pixels_with_channels_at_back() * 255).astype(np.uint8))
                    fitting = deep3dmodel.recontruct(im_menpo2PIL(img), lms)
                    img = menpo.image.Image(fitting['input'])

                final_uv, results_dict = operator.run(img, fitting)
                mio.export_image(final_uv, save_path.replace(ext,'_uv.'+ext))
                if args.frontalize:
                    mio.export_image(results_dict['frontal'], save_path.replace(ext,'_frontal.'+ext))
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
