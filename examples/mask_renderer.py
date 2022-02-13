import os, sys, datetime
import numpy as np
import os.path as osp
import cv2
import insightface
from insightface.app import MaskRenderer
from insightface.data import get_image as ins_get_image


if __name__ == "__main__":
    #make sure that you have download correct insightface model pack.
    #make sure that BFM.mat and BFM_UV.mat have been generated
    tool = MaskRenderer()
    tool.prepare(ctx_id=0, det_size=(128,128))
    image = ins_get_image('Tom_Hanks_54745')
    mask_image  = "mask_blue"
    params = tool.build_params(image)
    mask_out = tool.render_mask(image, mask_image, params)

    cv2.imwrite('output_mask.jpg', mask_out)


