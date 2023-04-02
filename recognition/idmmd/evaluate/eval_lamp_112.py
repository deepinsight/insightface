import numpy as np
import os,sys
sys.path.append(os.getcwd())
import argparse

from PIL import Image

import torch

from network.lightcnn112 import LightCNN_29Layers
from evaluate import evaluate2

fars = [10 ** -4, 10 ** -3, 10 ** -2]

parser = argparse.ArgumentParser()
parser.add_argument('--test_fold_id', default=1, type=int)
parser.add_argument('--input_mode', default='grey', choices=['grey'], type=str)
parser.add_argument('--model_mode', default='29', choices=['29'], type=str)
parser.add_argument('--model_name', default='', type=str)
parser.add_argument('--img_root', default='', type=str)
parser.add_argument('--test_mode', default='pretrain', type=str)
args = parser.parse_args()

INPUT_MODE = args.input_mode
MODEL_MODE = args.model_mode
model_name = args.model_name
test_mode = args.test_mode
img_root = args.img_root

tfi = args.test_fold_id
num_classes = 725
test_list_dir = './data/lamp/'
model_dir = f'./models/{test_mode}/'
model_path = os.path.join(model_dir, model_name)

def load_model(model, pretrained):
    weights = torch.load(pretrained)
    weights = weights['state_dict']

    model_dict = model.state_dict()

    weights = {k.replace('module.',''): v for k, v in weights.items() if k.replace('module.','') in model_dict.keys()}
    print("==> len of weights to be loaded: {}. \n".format(len(weights)))

    model.load_state_dict(weights, strict=False)
    model.eval()

def get_vis_nir_info(test_fold_id):
    def get_data(test_fold_id, mode='vis'):
        name = 'gallery_vis%d.txt' % (test_fold_id) if mode=='vis' else 'probe_nir%d.txt' % (test_fold_id)
        file_data = np.genfromtxt(test_list_dir + name , usecols=(0,1), skip_header=1, dtype=str)
        paths = file_data[:,0]

        # paths = [p for p in paths if os.path.exists(img_root + p)]
        return paths

    vis = get_data(test_fold_id, mode='vis')
    nir = get_data(test_fold_id, mode='nir')

    return vis, nir


class Embedding:
    def __init__(self, root, model):
        self.model = model
        self.root = root

        self.image_size = (112, 112)
        self.batch_size = 1

    def get(self, img):
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 1*112*112
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 1, self.image_size[1], self.image_size[0]),
                              dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255)
        feat = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()

    def extract_feats_labels(self, data_list):
        img_feats = []
        for imgPath in data_list:
            
            img = Image.open(os.path.join(self.root, imgPath)).convert('L')
            img = np.array(img)
            img = img[..., np.newaxis]

            img_feats.append(self.forward_db(self.get(img)).flatten())
        
        img_feats = np.array(img_feats).astype(np.float32)

        img_input_feats = img_feats[:, 0:img_feats.shape[1] //2] + img_feats[:, img_feats.shape[1] // 2:]
        img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True)) 

        return img_input_feats



if MODEL_MODE == '29':
    model = LightCNN_29Layers(num_classes=num_classes)
model.cuda()

embedding = Embedding(img_root, model)

############### test pre-trained models
if tfi == -1:
    n_fold = 10
    acc_ = []
    tarfar_ = np.zeros((n_fold, 4))
    for tf in range(n_fold):
        load_model(embedding.model, model_path)
        vis, nir = get_vis_nir_info(tf+1)
        
        feat_vis = embedding.extract_feats_labels(vis)
        feat_nir = embedding.extract_feats_labels(nir)
        
        label_matrix = np.load(test_list_dir + 'binary_lable_matrix_%d.npy' % (tf+1))
        label_matrix = label_matrix.T

        print("*" * 16)
        print("Fold id ", tf+1)
        print("Model: ", model_path)
        print("[query] feat_nir.shape ",feat_nir.shape)
        print("[gallery] feat_vis.shape ",feat_vis.shape)
        print("*" * 16)

        acc, tarfar = evaluate2(feat_vis, feat_nir, label_matrix, fars=fars)
        
        acc_.append(acc[0])
        tarfar_[tf,...] = np.array(tarfar)
    
    print('\n')
    print("*" * 16)
    print("MEAN")
    print("*" * 16)
    
    print("Rank 1 = {:.3%}  +- {:.2%}".format(np.mean(acc_), np.std(acc_)))
    var_mean = tarfar_.mean(0)
    var_std = tarfar_.std(0)
    for fpr_iter in np.arange(len(fars)):
        print("TAR {:.3%} +- {:.2%} @ FAR {:.4%}".format(var_mean[fpr_iter], var_std[fpr_iter], fars[fpr_iter]))

else:
    load_model(embedding.model, model_path)

    vis, nir= get_vis_nir_info(tfi)

    feat_vis = embedding.extract_feats_labels(vis)
    feat_nir = embedding.extract_feats_labels(nir)

    label_matrix = np.load(test_list_dir + 'binary_lable_matrix_%d.npy' % (tfi))
    label_matrix = label_matrix.T

    print("*" * 16)
    print("Fold id ", tfi)
    print("[query] feat_nir.shape ",feat_nir.shape)
    print("[gallery] feat_vis.shape ",feat_vis.shape)
    print("*" * 16)

    acc, tarfar = evaluate2(feat_vis, feat_nir, label_matrix, fars=fars)