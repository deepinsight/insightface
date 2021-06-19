#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import timeit
import sklearn
import cv2
import sys
import argparse
import glob
import numpy.matlib
import heapq
import math
from datetime import datetime as dt

from sklearn import preprocessing
sys.path.append('./recognition')
from embedding import Embedding
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap


def read_template_subject_id_list(path):
    ijb_meta = np.loadtxt(path, dtype=str, skiprows=1, delimiter=',')
    templates = ijb_meta[:, 0].astype(np.int)
    subject_ids = ijb_meta[:, 1].astype(np.int)
    return templates, subject_ids


def read_template_media_list(path):
    ijb_meta = np.loadtxt(path, dtype=str)
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    pairs = np.loadtxt(path, dtype=str)
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


#def get_image_feature(feature_path, faceness_path):
#    img_feats = np.loadtxt(feature_path)
#    faceness_scores = np.loadtxt(faceness_path)
#    return img_feats, faceness_scores
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


def image2template_feature(img_feats=None,
                           templates=None,
                           medias=None,
                           choose_templates=None,
                           choose_ids=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates, indices = np.unique(choose_templates, return_index=True)
    unique_subjectids = choose_ids[indices]
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
                    np.mean(face_norm_feats[ind_m], 0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    template_norm_feats = template_feats / np.sqrt(
        np.sum(template_feats**2, -1, keepdims=True))
    return template_norm_feats, unique_templates, unique_subjectids


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


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = cPickle.load(fid)
    return img_feats


def evaluation(query_feats, gallery_feats, mask):
    Fars = [0.01, 0.1]
    print(query_feats.shape)
    print(gallery_feats.shape)

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)
    print(top_inds.shape)

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if j == mask[i]:
            correct_num += 1
    print("top1 = {}".format(correct_num / query_num))
    # calculate top5
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:5]
        if mask[i] in j:
            correct_num += 1
    print("top5 = {}".format(correct_num / query_num))
    # calculate 10
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:10]
        if mask[i] in j:
            correct_num += 1
    print("top10 = {}".format(correct_num / query_num))

    neg_pair_num = query_num * gallery_num - query_num
    print(neg_pair_num)
    required_topk = [math.ceil(query_num * x) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        pos_sims.append(top_sims[i, gt])
        top_sims[i, gt] = -2.0

    pos_sims = np.array(pos_sims)
    print(pos_sims.shape)
    neg_sims = top_sims[np.where(top_sims > -2.0)]
    print("neg_sims num = {}".format(len(neg_sims)))
    neg_sims = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    print("after sorting , neg_sims num = {}".format(len(neg_sims)))
    for far, pos in zip(Fars, required_topk):
        th = neg_sims[pos - 1]
        recall = np.sum(pos_sims > th) / query_num
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(
            far, recall, th))


def gen_mask(query_ids, reg_ids):
    mask = []
    for query_id in query_ids:
        pos = [i for i, x in enumerate(reg_ids) if query_id == x]
        if len(pos) != 1:
            raise RuntimeError(
                "RegIdsError with id = {}， duplicate = {} ".format(
                    query_id, len(pos)))
        mask.append(pos[0])
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do ijb 1n test')
    # general
    parser.add_argument('--model-prefix',
                        default='',
                        help='path to load model.')
    parser.add_argument('--model-epoch', default=1, type=int, help='')
    parser.add_argument('--gpu', default=7, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--job',
                        default='insightface',
                        type=str,
                        help='job name')
    parser.add_argument('--target',
                        default='IJBC',
                        type=str,
                        help='target, set to IJBC or IJBB')
    args = parser.parse_args()
    target = args.target
    model_path = args.model_prefix
    gpu_id = args.gpu
    epoch = args.model_epoch
    meta_dir = "%s/meta" % args.target  #meta root dir
    if target == 'IJBC':
        gallery_s1_record = "%s_1N_gallery_G1.csv" % (args.target.lower())
        gallery_s2_record = "%s_1N_gallery_G2.csv" % (args.target.lower())
    else:
        gallery_s1_record = "%s_1N_gallery_S1.csv" % (args.target.lower())
        gallery_s2_record = "%s_1N_gallery_S2.csv" % (args.target.lower())
    gallery_s1_templates, gallery_s1_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s1_record))
    print(gallery_s1_templates.shape, gallery_s1_subject_ids.shape)

    gallery_s2_templates, gallery_s2_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s2_record))
    print(gallery_s2_templates.shape, gallery_s2_templates.shape)

    gallery_templates = np.concatenate(
        [gallery_s1_templates, gallery_s2_templates])
    gallery_subject_ids = np.concatenate(
        [gallery_s1_subject_ids, gallery_s2_subject_ids])
    print(gallery_templates.shape, gallery_subject_ids.shape)

    media_record = "%s_face_tid_mid.txt" % args.target.lower()
    total_templates, total_medias = read_template_media_list(
        os.path.join(meta_dir, media_record))
    print("total_templates", total_templates.shape, total_medias.shape)
    #load image features
    start = timeit.default_timer()
    feature_path = ''  #feature path
    face_path = ''  #face path
    img_path = './%s/loose_crop' % target
    img_list_path = './%s/meta/%s_name_5pts_score.txt' % (target,
                                                          target.lower())
    #img_feats, faceness_scores = get_image_feature(feature_path, face_path)
    img_feats, faceness_scores = get_image_feature(img_path, img_list_path,
                                                   model_path, epoch, gpu_id)
    print('img_feats', img_feats.shape)
    print('faceness_scores', faceness_scores.shape)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                              img_feats.shape[1]))

    # compute template features from image features.
    start = timeit.default_timer()
    # ==========================================================
    # Norm feature before aggregation into template feature?
    # Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
    # ==========================================================
    use_norm_score = True  # if True, TestMode(N1)
    use_detector_score = True  # if True, TestMode(D1)
    use_flip_test = True  # if True, TestMode(F1)

    if use_flip_test:
        # concat --- F1
        #img_input_feats = img_feats
        # add --- F2
        img_input_feats = img_feats[:, 0:int(
            img_feats.shape[1] / 2)] + img_feats[:,
                                                 int(img_feats.shape[1] / 2):]
    else:
        img_input_feats = img_feats[:, 0:int(img_feats.shape[1] / 2)]

    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        # normalise features to remove norm information
        img_input_feats = img_input_feats / np.sqrt(
            np.sum(img_input_feats**2, -1, keepdims=True))

    if use_detector_score:
        img_input_feats = img_input_feats * np.matlib.repmat(
            faceness_scores[:, np.newaxis], 1, img_input_feats.shape[1])
    else:
        img_input_feats = img_input_feats
    print("input features shape", img_input_feats.shape)

    #load gallery feature
    gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = image2template_feature(
        img_input_feats, total_templates, total_medias, gallery_templates,
        gallery_subject_ids)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print("gallery_templates_feature", gallery_templates_feature.shape)
    print("gallery_unique_subject_ids", gallery_unique_subject_ids.shape)
    #np.savetxt("gallery_templates_feature.txt", gallery_templates_feature)
    #np.savetxt("gallery_unique_subject_ids.txt", gallery_unique_subject_ids)

    #load prope feature
    probe_mixed_record = "%s_1N_probe_mixed.csv" % target.lower()
    probe_mixed_templates, probe_mixed_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, probe_mixed_record))
    print(probe_mixed_templates.shape, probe_mixed_subject_ids.shape)
    probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = image2template_feature(
        img_input_feats, total_templates, total_medias, probe_mixed_templates,
        probe_mixed_subject_ids)
    print("probe_mixed_templates_feature", probe_mixed_templates_feature.shape)
    print("probe_mixed_unique_subject_ids",
          probe_mixed_unique_subject_ids.shape)
    #np.savetxt("probe_mixed_templates_feature.txt", probe_mixed_templates_feature)
    #np.savetxt("probe_mixed_unique_subject_ids.txt", probe_mixed_unique_subject_ids)

    #root_dir = "" #feature root dir
    #gallery_id_path = "" #id filepath
    #gallery_feats_path = "" #feature filelpath
    #print("{}: start loading gallery feat {}".format(dt.now(), gallery_id_path))
    #gallery_ids, gallery_feats = load_feat_file(root_dir, gallery_id_path, gallery_feats_path)
    #print("{}: end loading gallery feat".format(dt.now()))
    #
    #probe_id_path = "probe_mixed_unique_subject_ids.txt" #probe id filepath
    #probe_feats_path = "probe_mixed_templates_feature.txt" #probe feats filepath
    #print("{}: start loading probe feat {}".format(dt.now(), probe_id_path))
    #probe_ids, probe_feats = load_feat_file(root_dir, probe_id_path, probe_feats_path)
    #print("{}: end loading probe feat".format(dt.now()))

    gallery_ids = gallery_unique_subject_ids
    gallery_feats = gallery_templates_feature
    probe_ids = probe_mixed_unique_subject_ids
    probe_feats = probe_mixed_templates_feature

    mask = gen_mask(probe_ids, gallery_ids)

    print("{}: start evaluation".format(dt.now()))
    evaluation(probe_feats, gallery_feats, mask)
    print("{}: end evaluation".format(dt.now()))
