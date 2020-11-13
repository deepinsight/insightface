import numpy as np
import math
import cv2
from skimage import transform as stf


def transform(data, center, output_size, scale, rotation):
    scale_ratio = float(output_size) / scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = stf.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = stf.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = stf.SimilarityTransform(rotation=rot)
    t4 = stf.SimilarityTransform(translation=(output_size / 2,
                                              output_size / 2))
    t = t1 + t2 + t3 + t4
    trans = t.params[0:2]
    #print('M', scale, rotation, trans)
    cropped = cv2.warpAffine(data,
                             trans, (output_size, output_size),
                             borderValue=0.0)
    return cropped, trans


def transform_pt(pt, trans):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(trans, new_pt)
    #print('new_pt', new_pt.shape, new_pt)
    return new_pt[:2]


def gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    assert (sigma >= 0)
    if sigma == 0:
        img[pt[1], pt[0]] = 1.0
        return True
    #assert pt[0]<=img.shape[1]
    #assert pt[1]<=img.shape[0]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0
            or br[1] < 0):
        # If not, just return the image as is
        #print('gaussian error')
        return False
        #return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return True
    #return img


def estimate_trans_bbox(face, input_size, s=2.0):
    w = face[2] - face[0]
    h = face[3] - face[1]
    wc = int((face[2] + face[0]) / 2)
    hc = int((face[3] + face[1]) / 2)
    im_size = max(w, h)
    #size = int(im_size*1.2)
    scale = input_size / (max(w, h) * s)
    M = [
        [scale, 0, input_size / 2 - wc * scale],
        [0, scale, input_size / 2 - hc * scale],
    ]
    M = np.array(M)
    return M
