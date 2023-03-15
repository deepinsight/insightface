import numpy as np
import pandas as pd
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

print("*" * 16)
print("INPUT_MODE: ", INPUT_MODE)
print("MODEL_MODE: ", MODEL_MODE)
print("model name: ", model_name)
print("*" * 16)

tfi = args.test_fold_id
num_classes = 725
test_list_dir = 'data/casia/'
model_dir = f'./models/{test_mode}/'
model_path = os.path.join(model_dir, model_name)

def load_model(model, pretrained):
    weights = torch.load(pretrained)
    weights = weights['state_dict']

    model_dict = model.state_dict()
    
    weights = {k.replace('module.',''): v for k, v in weights.items() if k.replace('module.','') in model_dict.keys() and 'fc2' not in k}
    print("==> len of weights to be loaded: {}. \n".format(len(weights)))
    model.load_state_dict(weights, strict=False)
    model.eval()


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
        pids = []
        for (imgPath, pid) in data_list:
            img = Image.open(os.path.join(self.root, imgPath)).convert('L')

            img = np.array(img)
            img = img[..., np.newaxis]

            img_feats.append(self.forward_db(self.get(img)).flatten())
            pids.append(pid)
        
        img_feats = np.array(img_feats).astype(np.float32)
        img_input_feats = img_feats[:, 0:img_feats.shape[1] //2] + img_feats[:, img_feats.shape[1] // 2:]
        img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True)) 
        pids = np.array(pids)

        return img_input_feats, pids


def get_vis_nir_info(test_fold_id):
    vis = pd.read_csv(os.path.join(test_list_dir, 'vis_gallery_%d.txt' % test_fold_id), header=None, sep=' ')
    vis_labels = [int(s.split('\\')[-2]) for s in vis[0]]
    vis = vis[0].apply(lambda s: rename_path(s)).tolist()

    nir = pd.read_csv(os.path.join(test_list_dir, 'nir_probe_%d.txt' % test_fold_id), header=None, sep=' ')
    nir_labels = [int(s.split('\\')[-2]) for s in nir[0]]
    nir = nir[0].apply(lambda s: rename_path(s)).tolist()

    vis = [(p,l) for (p,l) in zip(vis, vis_labels)]
    nir = [(p,l) for (p,l) in zip(nir, nir_labels)]

    return vis,nir

def rename_path(s):
    """messy path names, inconsistency between 10-folds and how data are actually saved"""
    s = s.split(".")[0]
    gr, mod, id, img = s.split("\\")
    ext = 'jpg' if (mod == 'VIS') else 'bmp'
    return "%s/%s_%s_%s_%s.%s" % (mod, gr, mod, id, img, ext)



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
        vis, nir= get_vis_nir_info(tf+1)
        
        feat_vis, label_vis = embedding.extract_feats_labels(vis)
        feat_nir, label_nir = embedding.extract_feats_labels(nir)

        labels = np.equal.outer(label_vis, label_nir).astype(np.float32)

        print("*" * 16)
        print("Fold id ", tf+1)
        print("Model: ", model_path)
        print("[query] feat_nir.shape ",feat_nir.shape)
        print("[gallery] feat_vis.shape ",feat_vis.shape)
        print("*" * 16)

        acc, tarfar = evaluate2(feat_vis, feat_nir, labels, fars=fars)
        
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
    if not os.path.exists(model_path):
        print("cannot find model ",model_path)
        sys.exit()

    model = load_model(embedding.model, model_path)

    vis, nir= get_vis_nir_info(tfi)

    feat_vis, label_vis = embedding.extract_feats_labels(vis)
    feat_nir, label_nir = embedding.extract_feats_labels(nir)

    labels = np.equal.outer(label_vis, label_nir).astype(np.float32)

    print("*" * 16)
    print("Fold id ", tfi)
    print("[query] feat_nir.shape ",feat_nir.shape)
    print("[gallery] feat_vis.shape ",feat_vis.shape)
    print("*" * 16)

    acc, tarfar = evaluate2(feat_vis, feat_nir, labels, fars=fars)

