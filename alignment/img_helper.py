import numpy as np
import scipy.misc
import scipy.signal
import math

#import draw
#import ref

# =============================================================================
# General image processing functions
# =============================================================================

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    #h = 200 * scale
    #h = 100 * scale
    h = scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    #print('new_pt', new_pt.shape, new_pt)
    return new_pt[:2].astype(int)

def crop_center(img,crop_size):
    y,x = img.shape[0], img.shape[1]
    startx = x//2-(crop_size[1]//2)
    starty = y//2-(crop_size[0]//2)    
    #print(startx, starty, crop_size)
    return img[starty:(starty+crop_size[0]),startx:(startx+crop_size[1]),:]

def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)
    #print('new_img', new_img.shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        #print('before rotate', new_img.shape, rot)
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    return scipy.misc.imresize(new_img, res)

def crop2(img, center, scale, res, rot=0):
    # Upper left point
    rad = np.min( [center[0], img.shape[0] - center[0], center[1], img.shape[1] - center[1]] )
    new_img = img[(center[0]-rad):(center[0]+rad),(center[1]-rad):(center[1]+rad),:]
    #print('new_img', new_img.shape)
    if not rot == 0:
        new_img = scipy.misc.imrotate(new_img, rot)
    new_img = crop_center(new_img, (scale,scale))
    return scipy.misc.imresize(new_img, res)

def nms(img):
    # Do non-maximum suppression on a 2D array
    win_size = 3
    domain = np.ones((win_size, win_size))
    maxes = scipy.signal.order_filter(img, domain, win_size ** 2 - 1)
    diff = maxes - img
    result = img.copy()
    result[diff > 0] = 0
    return result


def gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    assert(sigma>0)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
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
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return True
    #return img

