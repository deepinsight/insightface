import numpy as np
import math
import cv2

# Adapted from: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
def _gaussian(size=3, sigma=0.25, amplitude=1, normalize=False, width=None, height=None, sigma_horz=None,
              sigma_vert=None, mean_horz=0.5, mean_vert=0.5):
    """ Generate a guassian kernel.

    Args:
        size (int): The size of the kernel if the width or height are not specified
        sigma (float): Standard deviation of the kernel if sigma_horz or sigma_vert are not specified
        amplitude: The scale of the kernel
        normalize: If True, the kernel will be normalized such as values will sum to one
        width (int, optional): The width of the kernel
        height (int, optional): The height of the kernel
        sigma_horz (float, optional): Horizontal standard deviation of the kernel
        sigma_vert (float, optional): Vertical standard deviation of the kernel
        mean_horz (float): Horizontal mean of the kernel
        mean_vert (float): Vertical mean of the kernel

    Returns:
        np.array: The computed gaussian kernel
    """
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                    sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)

    return gauss


# Adapted from: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
def draw_gaussian(image, point, sigma):
    """ Draw gaussian circle at a point in an image.

    Args:
        image (np.array): An image of shape (H, W)
        point (np.array): The center point of the guassian circle
        sigma (float): Standard deviation of the gaussian kernel

    Returns:
        np.array: The image with the drawn gaussian.
    """
    # Check if the gaussian is inside
    point[0] = round(point[0], 2)
    point[1] = round(point[1], 2)

    ul = [math.floor(point[0] - 7.5 * sigma), math.floor(point[1] - 7.5 * sigma)]
    br = [math.floor(point[0] + 7.5 * sigma), math.floor(point[1] + 7.5 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 15 * sigma + 1
    g = _gaussian(size, sigma=0.1)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] = \
        image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1

    return image


# Adapted from: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/api.py
def generate_heatmaps(height, width, points, sigma=None):
    """ Generate heatmaps corresponding to a set of points.

    Args:
        height (int): Heatmap height
        width (int): Heatmap width
        points (np.array): An array of points of shape (N, 2)
        sigma (float, optional): Standard deviation of the gaussian kernel. If not specified it will be determined
            from the width of the heatmap

    Returns:
        np.array: The generated heatmaps.
    """
    sigma = max(1, int(np.round(width / 128.))) if sigma is None else sigma
    heatmaps = np.zeros((points.shape[0], height, width), dtype=np.float32)
    for i in range(points.shape[0]):
        if points[i, 0] > 0:
            heatmaps[i] = draw_gaussian(
                heatmaps[i], points[i], sigma)

    return heatmaps



if __name__ == "__main__":

    #you can use [X,2] matrix
    points = np.array([
        [(30.2946)+8, 51.6963],
        [(65.5318)+8, 51.5014],
        [(48.0252)+8, 71.7366],
        [(33.5493)+8, 92.3655],
        [(62.7299)+8, 92.2041]], dtype=np.float32)


    heatmaps = generate_heatmaps(width = 112,
                                 height = 112,
                                 points = points,
                                 sigma = 3)
    print(heatmaps.shape)

    final_heatmap = np.sum(heatmaps, axis=0)
    cv2.imwrite("final_heatmap.png", final_heatmap*255)
    print("end")