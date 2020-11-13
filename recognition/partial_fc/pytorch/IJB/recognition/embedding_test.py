import argparse
import cv2
import numpy as np
import sys
import mxnet as mx
import datetime
from skimage import transform as trans
import sklearn
from sklearn import preprocessing
import torch
from torchvision import transforms
sys.path.append('/root/xy/work_dir/xyface/')
from backbones import iresnet50, iresnet100
from torch.nn.parallel import DistributedDataParallel


class Embedding:
    def __init__(self, prefix, epoch, data_shape, batch_size=1, ctx_id=0):
        print('loading', prefix, epoch)
        image_size = (112, 112)
        self.image_size = image_size
        weight = torch.load(prefix)
        resnet = iresnet50().cuda()
        resnet.load_state_dict(weight)
        model = torch.nn.DataParallel(resnet)
        self.model = model
        self.model.eval()
        src = np.array(
            [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
             [33.5493, 92.3655], [62.7299, 92.2041]],
            dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]),
                              dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()


if __name__ == "__main__":
    weight = torch.load('/root/xy/work_dir/xyface/backbone0.pth')
    resnet = resnet50().cuda()
    resnet.load_state_dict(weight)
    res = torch.nn.DataParallel(resnet, [0, 1, 2, 3, 4, 5, 6, 7])
    tin = torch.Tensor(1023, 3, 112, 112).cuda()
    out = res(tin)
    print(out.size())
