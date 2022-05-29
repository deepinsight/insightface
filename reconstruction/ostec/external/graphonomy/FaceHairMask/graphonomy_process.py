import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms


class Scale_only_img(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        img = F.interpolate(  # NOTE: requires 4D
            img, scale_factor=self.scale, mode="bilinear"
        )
        # print("Scale:", img.shape, img.min(), img.max())
        # import ipdb; ipdb.set_trace()
        return {"image": img, "label": mask}


class Normalize_xception_tf_only_img(object):
    def __call__(self, sample):
        img = sample["image"]
        img = (img * 2.0) / 255.0 - 1
        # print("Normalize:", img.shape, img.min(), img.max())
        # import ipdb; ipdb.set_trace()
        return {"image": img, "label": sample["label"]}


class ToTensor_only_img(object):
    def __init__(self):
        self.rgb2bgr = transforms.Lambda(lambda x: x[[2, 1, 0], ...])

    def __call__(self, sample):
        # sample: N x C x H x W
        img = sample["image"]
        img = torch.squeeze(img, axis=0)
        # sample: C x H x W
        img = self.rgb2bgr(img)
        # print("To Tensor:", img.shape, img.min(), img.max())
        # img = torch.unsqueeze(img, axis=0)
        # import ipdb; ipdb.set_trace()
        return {"image": img, "label": sample["label"]}


class HorizontalFlip_only_img(object):
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        img = torch.flip(img, [-1])
        # print("Horizontal:", img.shape, img.min(), img.max())
        # import ipdb; ipdb.set_trace()
        return {"image": img, "label": mask}
