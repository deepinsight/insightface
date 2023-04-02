import numpy as np
import pandas as pd
import os,sys
sys.path.append(os.getcwd())
print(sys.path)
import argparse
import torch

from PIL import Image

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

num_classes = 725
test_list_dir = './data/oulu/' 
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


def get_vis_nir_info_csv():
    vis = pd.read_csv(test_list_dir + 'vis_test_paths.csv', header=None, sep=' ')
    vis_labels = [int(s.strip().split(',')[-1].split('P')[-1]) for s in vis[0]]
    vis = [s.strip().split(',')[0] for s in vis[0]]

    nir = pd.read_csv(test_list_dir + 'nir_test_paths.csv', header=None, sep=' ')
    nir_labels = [int(s.strip().split(',')[-1].split('P')[-1]) for s in nir[0]]
    nir = [s.strip().split(',')[0] for s in nir[0]]

    vis = [(p,l) for (p,l) in zip(vis, vis_labels)]
    nir = [(p,l) for (p,l) in zip(nir, nir_labels)]
    
    return vis,nir

def get_vis_nir_info_txt():
    def read_file(file_name):
        with open(test_list_dir + file_name, 'r') as f:
            lines = f.readlines()
        paths = [s.strip().split(' ')[0] for s in lines]
        labels = [int(s.strip().split(' ')[1]) for s in lines]
        info = [(p,l) for (p,l) in zip(paths, labels)]

        return info

    vis = read_file('test_vis_paths.txt')
    nir = read_file('test_nir_paths.txt')
    
    return vis, nir


### Testing pretrain/finetune model
if test_mode == 'pretrain':
    vis, nir = get_vis_nir_info_csv()
elif test_mode == "finetune":
    vis, nir = get_vis_nir_info_txt()
else:
    print("Wrong test_mode!!!")

if MODEL_MODE == '29':
    model = LightCNN_29Layers(num_classes=num_classes)

model.cuda()

embedding = Embedding(img_root, model)

if not os.path.exists(model_path):
    print("cannot find model ",model_path)
    sys.exit()

load_model(embedding.model, model_path)

feat_vis, label_vis = embedding.extract_feats_labels(vis)
feat_nir, label_nir = embedding.extract_feats_labels(nir)

labels = np.equal.outer(label_vis, label_nir).astype(np.float32)

print("*" * 16)
print("INPUT_MODE: ", INPUT_MODE)
print("MODEL_MODE: ", MODEL_MODE)
print("model path: ", model_path)
print("*" * 16)
print("[query] feat_nir.shape ",feat_nir.shape)
print("[gallery] feat_vis.shape ",feat_vis.shape)
print("*" * 16)

acc, tarfar = evaluate2(feat_vis, feat_nir, labels, fars=fars)