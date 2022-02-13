import cv2
import os
import os.path as osp
from pathlib import Path

class ImageCache:
    data = {}

def get_image(name, to_rgb=False):
    key = (name, to_rgb)
    if key in ImageCache.data:
        return ImageCache.data[key]
    images_dir = osp.join(Path(__file__).parent.absolute(), 'images')
    ext_names = ['.jpg', '.png', '.jpeg']
    image_file = None
    for ext_name in ext_names:
        _image_file = osp.join(images_dir, "%s%s"%(name, ext_name))
        if osp.exists(_image_file):
            image_file = _image_file
            break
    assert image_file is not None, '%s not found'%name
    img = cv2.imread(image_file)
    if to_rgb:
        img = img[:,:,::-1]
    ImageCache.data[key] = img
    return img

