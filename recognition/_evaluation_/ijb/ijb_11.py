# coding: utf-8

import os
import numpy as np
#import cPickle
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import cv2
import sys
import glob
sys.path.append('./recognition')
from embedding import Embedding
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--model-epoch', default=1, type=int, help='')
parser.add_argument('--gpu', default=7, type=int, help='gpu id')
parser.add_argument('--batch-size', default=32, type=int, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target',
                    default='IJBC',
                    type=str,
                    help='target, set to IJBC or IJBB')
args = parser.parse_args()

target = args.target
model_path = args.model_prefix
gpu_id = args.gpu
epoch = args.model_epoch
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job


def read_template_media_list(path):
    #ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int32)
    medias = ijb_meta[:, 2].astype(np.int32)
    return templates, medias


# In[ ]:


def read_template_pair_list(path):
    #pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    #print(pairs.shape)
    #print(pairs[:, 0].astype(np.int32))
    t1 = pairs[:, 0].astype(np.int32)
    t2 = pairs[:, 1].astype(np.int32)
    label = pairs[:, 2].astype(np.int32)
    return t1, t2, label


# In[ ]:


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# In[ ]:


def get_image_feature(img_path, img_list_path, model_path, epoch, gpu_id):
    img_list = open(img_list_path)
    embedding = Embedding(model_path, epoch, gpu_id)
    files = img_list.readlines()
    print('files:', len(files))
    faceness_scores = []
    img_feats = []
    for img_index, each_line in enumerate(files):
        if img_index % 500 == 0:
            print('processing', img_index)
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        img_feats.append(embedding.get(img, lmk))
        faceness_scores.append(name_lmk_score[-1])
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)

    #img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
    #faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
    return img_feats, faceness_scores


# In[ ]:


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
    #template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    #print(template_norm_feats.shape)
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

# In[ ]:

assert target == 'IJBC' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % target, '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % target,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# In[ ]:

# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
img_path = './%s/loose_crop' % target
img_list_path = './%s/meta/%s_name_5pts_score.txt' % (target, target.lower())
img_feats, faceness_scores = get_image_feature(img_path, img_list_path,
                                               model_path, epoch, gpu_id)
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
    #img_input_feats = img_feats
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
    #img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:,np.newaxis], 1, img_input_feats.shape[1])
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

save_path = './%s_result' % target

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
#x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
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
        #tpr_fpr_row.append('%.4f' % tpr[min_index])
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
#plt.show()
fig.savefig(os.path.join(save_path, '%s.pdf' % job))
print(tpr_fpr_table)
