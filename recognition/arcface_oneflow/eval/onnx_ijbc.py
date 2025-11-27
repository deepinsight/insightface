import argparse
import os
import pickle
import timeit

import cv2
import mxnet as mx
import numpy as np
import pandas as pd
import prettytable
import skimage.transform
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize

from onnx_helper import ArcFaceORT

SRC = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)
SRC[:, 0] += 8.0


class AlignedDataSet(mx.gluon.data.Dataset):
    def __init__(self, root, lines, align=True):
        self.lines = lines
        self.root = root
        self.align = align

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        each_line = self.lines[idx]
        name_lmk_score = each_line.strip().split(" ")
        name = os.path.join(self.root, name_lmk_score[0])
        img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        landmark5 = np.array(
            [float(x) for x in name_lmk_score[1:-1]], dtype=np.float32
        ).reshape((5, 2))
        st = skimage.transform.SimilarityTransform()
        st.estimate(landmark5, SRC)
        img = cv2.warpAffine(img, st.params[0:2, :], (112, 112), borderValue=0.0)
        img_1 = np.expand_dims(img, 0)
        img_2 = np.expand_dims(np.fliplr(img), 0)
        output = np.concatenate((img_1, img_2), axis=0).astype(np.float32)
        output = np.transpose(output, (0, 3, 1, 2))
        output = mx.nd.array(output)
        return output


def extract(model_root, dataset):
    model = ArcFaceORT(model_path=model_root)
    model.check()
    feat_mat = np.zeros(shape=(len(dataset), 2 * model.feat_dim))

    def batchify_fn(data):
        return mx.nd.concat(*data, dim=0)

    data_loader = mx.gluon.data.DataLoader(
        dataset,
        128,
        last_batch="keep",
        num_workers=4,
        thread_pool=True,
        prefetch=16,
        batchify_fn=batchify_fn,
    )
    num_iter = 0
    for batch in data_loader:
        batch = batch.asnumpy()
        batch = (batch - model.input_mean) / model.input_std
        feat = model.session.run(model.output_names, {model.input_name: batch})[0]
        feat = np.reshape(feat, (-1, model.feat_dim * 2))
        feat_mat[128 * num_iter : 128 * num_iter + feat.shape[0], :] = feat
        num_iter += 1
        if num_iter % 50 == 0:
            print(num_iter)
    return feat_mat


def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=" ", header=None).values
    templates = ijb_meta[:, 1].astype(np.int32)
    medias = ijb_meta[:, 2].astype(np.int32)
    return templates, medias


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=" ", header=None).values
    t1 = pairs[:, 0].astype(np.int32)
    t2 = pairs[:, 1].astype(np.int32)
    label = pairs[:, 2].astype(np.int32)
    return t1, t2, label


def read_image_feature(path):
    with open(path, "rb") as fid:
        img_feats = pickle.load(fid)
    return img_feats


def image2template_feature(img_feats=None, templates=None, medias=None):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True),
                ]
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print("Finish Calculating {} template features.".format(count_template))
    template_norm_feats = normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000
    sublists = [total_pairs[i : i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print("Finish {}/{} pairs.".format(c, total_sublists))
    return score


def verification2(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    # small batchsize instead of all pairs in one batch due to the memory limiation
    batchsize = 100000
    sublists = [total_pairs[i : i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print("Finish {}/{} pairs.".format(c, total_sublists))
    return score


def main(args):
    use_norm_score = True  # if Ture, TestMode(N1)
    use_detector_score = True  # if Ture, TestMode(D1)
    use_flip_test = True  # if Ture, TestMode(F1)
    assert args.target == "IJBC" or args.target == "IJBB"

    start = timeit.default_timer()
    templates, medias = read_template_media_list(
        os.path.join(
            "%s/meta" % args.image_path, "%s_face_tid_mid.txt" % args.target.lower()
        )
    )
    stop = timeit.default_timer()
    print("Time: %.2f s. " % (stop - start))

    start = timeit.default_timer()
    p1, p2, label = read_template_pair_list(
        os.path.join(
            "%s/meta" % args.image_path,
            "%s_template_pair_label.txt" % args.target.lower(),
        )
    )
    stop = timeit.default_timer()
    print("Time: %.2f s. " % (stop - start))

    start = timeit.default_timer()
    img_path = "%s/loose_crop" % args.image_path
    img_list_path = "%s/meta/%s_name_5pts_score.txt" % (
        args.image_path,
        args.target.lower(),
    )
    img_list = open(img_list_path)
    files = img_list.readlines()
    dataset = AlignedDataSet(root=img_path, lines=files, align=True)
    img_feats = extract(args.model_root, dataset)

    faceness_scores = []
    for each_line in files:
        name_lmk_score = each_line.split()
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    stop = timeit.default_timer()
    print("Time: %.2f s. " % (stop - start))
    print("Feature Shape: ({} , {}) .".format(img_feats.shape[0], img_feats.shape[1]))
    start = timeit.default_timer()

    if use_flip_test:
        img_input_feats = (
            img_feats[:, 0 : img_feats.shape[1] // 2]
            + img_feats[:, img_feats.shape[1] // 2 :]
        )
    else:
        img_input_feats = img_feats[:, 0 : img_feats.shape[1] // 2]

    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        img_input_feats = img_input_feats / np.sqrt(
            np.sum(img_input_feats ** 2, -1, keepdims=True)
        )

    if use_detector_score:
        print(img_input_feats.shape, faceness_scores.shape)
        img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
    else:
        img_input_feats = img_input_feats

    template_norm_feats, unique_templates = image2template_feature(
        img_input_feats, templates, medias
    )
    stop = timeit.default_timer()
    print("Time: %.2f s. " % (stop - start))

    start = timeit.default_timer()
    score = verification(template_norm_feats, unique_templates, p1, p2)
    stop = timeit.default_timer()
    print("Time: %.2f s. " % (stop - start))
    save_path = os.path.join(args.result_dir, "{}_result".format(args.target))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    score_save_file = os.path.join(save_path, "{}.npy".format(args.model_root))
    np.save(score_save_file, score)
    files = [score_save_file]
    methods = []
    scores = []
    for file in files:
        methods.append(os.path.basename(file))
        scores.append(np.load(file))
    methods = np.array(methods)
    scores = dict(zip(methods, scores))
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = prettytable.PrettyTable(["Methods"] + [str(x) for x in x_labels])
    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)
        tpr_fpr_row = []
        tpr_fpr_row.append("%s-%s" % (method, args.target))
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(
                list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr))))
            )
            tpr_fpr_row.append("%.2f" % (tpr[min_index] * 100))
        tpr_fpr_table.add_row(tpr_fpr_row)
    print(tpr_fpr_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do ijb test")
    # general
    parser.add_argument("--model-root", default="", help="path to load model.")
    parser.add_argument("--image-path", default="", type=str, help="")
    parser.add_argument("--result-dir", default=".", type=str, help="")
    parser.add_argument(
        "--target", default="IJBC", type=str, help="target, set to IJBC or IJBB"
    )
    main(parser.parse_args())
