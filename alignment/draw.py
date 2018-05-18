import numpy as np
import skimage.draw

def line(img, pt1, pt2, color, width):
    # Draw a line on an image
    # Make sure dimension of color matches number of channels in img

    # First get coordinates for corners of the line
    diff = np.array([pt1[1] - pt2[1], pt1[0] - pt2[0]], np.float)
    mag = np.linalg.norm(diff)
    if mag >= 1:
        diff *= width / (2 * mag)
        x = np.array([pt1[0] - diff[0], pt2[0] - diff[0], pt2[0] + diff[0], pt1[0] + diff[0]], int)
        y = np.array([pt1[1] + diff[1], pt2[1] + diff[1], pt2[1] - diff[1], pt1[1] - diff[1]], int)
    else:
        d = float(width) / 2
        x = np.array([pt1[0] - d, pt1[0] + d, pt1[0] + d, pt1[0] - d], int)
        y = np.array([pt1[1] - d, pt1[1] - d, pt1[1] + d, pt1[1] + d], int)

    # noinspection PyArgumentList
    rr, cc = skimage.draw.polygon(y, x, img.shape)
    img[rr, cc] = color

    return img

def limb(img, pt1, pt2, color, width):
    # Specific handling of a limb, in case the annotation isn't there for one of the joints
    if pt1[0] > 0 and pt2[0] > 0:
        line(img, pt1, pt2, color, width)
    elif pt1[0] > 0:
        circle(img, pt1, color, width)
    elif pt2[0] > 0:
        circle(img, pt2, color, width)

def gaussian(img, pt, sigma):
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

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
    return img

def circle(img, pt, color, radius):
    # Draw a circle
    # Mostly a convenient wrapper for skimage.draw.circle

    rr, cc = skimage.draw.circle(pt[1], pt[0], radius, img.shape)
    img[rr, cc] = color
    return img
