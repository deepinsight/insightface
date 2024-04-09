import argparse
import os
import pickle
import timeit
import warnings
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
import sklearn
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from mxnet.gluon.data import Dataset, DataLoader
from prettytable import PrettyTable
from skimage import transform as trans
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--model-epoch', default=1, type=int, help='')
parser.add_argument('--image-path', default='', type=str, help='')
parser.add_argument('--result-dir', default='.', type=str, help='')
parser.add_argument('--gpu', default='0', type=str, help='gpu id')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('-es', '--emb-size', type=int, help='embedding size')
parser.add_argument('--target',
                    default='IJBC',
                    type=str,
                    help='target, set to IJBC or IJBB')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

target = args.target
model_path = args.model_prefix
image_path = args.image_path
result_dir = args.result_dir
epoch = args.model_epoch
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
batch_size = args.batch_size


class DatasetIJB(Dataset):
    def __init__(self, root, lines, align=True):
        self.src = np.array(
            [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
             [33.5493, 92.3655], [62.7299, 92.2041]],
            dtype=np.float32)
        self.src[:, 0] += 8.0
        self.lines = lines
        self.img_root = root
        self.align = align

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        each_line = self.lines[idx]
        name_lmk_score = each_line.strip().split(' ')  # "name lmk score"
        img_name = os.path.join(self.img_root, name_lmk_score[0])
        img = cv2.imread(img_name)

        if self.align:
            landmark = np.array([float(x) for x in name_lmk_score[1:-1]],
                                dtype=np.float32)
            landmark = landmark.reshape((5, 2))
            #
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
            #
            tform = trans.SimilarityTransform()
            tform.estimate(landmark5, self.src)
            #
            M = tform.params[0:2, :]
            img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, 112, 112), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return mx.nd.array(input_blob)


def extract_parallel(prefix, epoch, dataset, batch_size, size):
    # init
    model_list = list()
    num_ctx = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
    num_iter = 0
    feat_mat = mx.nd.zeros(shape=(len(dataset), 2 * size))

    def batchify_fn(data):
        return mx.nd.concat(*data, dim=0)

    data_loader = DataLoader(dataset,
                             batch_size,
                             last_batch='keep',
                             num_workers=8,
                             thread_pool=True,
                             prefetch=16,
                             batchify_fn=batchify_fn)
    symbol, arg_params, aux_params = mx.module.module.load_checkpoint(
        prefix, epoch)
    all_layers = symbol.get_internals()
    symbol = all_layers['fc1_output']

    # init model list
    for i in range(num_ctx):
        model = mx.mod.Module(symbol, context=mx.gpu(i), label_names=None)
        model.bind(for_training=False,
                   data_shapes=[('data', (2 * batch_size, 3, 112, 112))])
        model.set_params(arg_params, aux_params)
        model_list.append(model)

    # extract parallel and async
    num_model = len(model_list)
    for image in tqdm(data_loader):
        data_batch = mx.io.DataBatch(data=(image, ))
        model_list[num_iter % num_model].forward(data_batch, is_train=False)
        feat = model_list[num_iter %
                          num_model].get_outputs(merge_multi_context=True)[0]
        feat = mx.nd.L2Normalization(feat)
        feat = mx.nd.reshape(feat, (-1, size * 2))
        feat_mat[batch_size * num_iter:batch_size * num_iter +
                 feat.shape[0], :] = feat.as_in_context(mx.cpu())
        num_iter += 1
        #if num_iter % 20 == 0:
        #    mx.nd.waitall()
    return feat_mat.asnumpy()


# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int32)
    medias = ijb_meta[:, 2].astype(np.int32)
    return templates, medias


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int32)
    t2 = pairs[:, 1].astype(np.int32)
    label = pairs[:, 2].astype(np.int32)
    return t1, t2, label


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t, ) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m, ) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


# In[ ]:


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


# In[ ]:
def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1), ))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# # Step1: Load Meta Data

assert target == 'IJBC' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % image_path,
                 '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
img_path = '%s/loose_crop' % image_path
img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_list = open(img_list_path)
files = img_list.readlines()
dataset = DatasetIJB(root=img_path, lines=files, align=True)
img_feats = extract_parallel(args.model_prefix,
                             args.model_epoch,
                             dataset,
                             args.batch_size,
                             size=args.emb_size)

faceness_scores = []
for each_line in files:
    name_lmk_score = each_line.split()
    faceness_scores.append(name_lmk_score[-1])

faceness_scores = np.array(faceness_scores).astype(np.float32)

stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                          img_feats.shape[1]))

# # Step3: Get Template Features

# In[ ]:

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats**2, -1, keepdims=True))

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    # img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:,np.newaxis], 1, img_input_feats.shape[1])
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 4: Get Template Similarity Scores

# In[ ]:

# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

save_path = result_dir + '/%s_result' % target

if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % job)
np.save(score_save_file, score)

# # Step 5: Get ROC Curves and TPR@FPR Table

# In[ ]:

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
# x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        # tpr_fpr_row.append('%.4f' % tpr[min_index])
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10**-6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
# plt.show()
fig.savefig(os.path.join(save_path, '%s.pdf' % job))
print(tpr_fpr_table)
