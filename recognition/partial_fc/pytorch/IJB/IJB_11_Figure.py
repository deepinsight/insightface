import os
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap

target = 'IJBC'
job = 'IJBC'
title = 'IJB-C'

root = '/train/trainset/1'
score_save_file_1 = '{}/glint-face/IJB/result/celeb360kfinal0.1/{}_result/cosface.npy'.format(
    root, target)
score_save_file_2 = '{}/glint-face/IJB/result/celeb360kfinal1.0/{}_result/cosface.npy'.format(
    root, target)
score_save_file_3 = '{}/glint-face/IJB/result/emore0.4/{}_result/cosface.npy'.format(
    root, target)
score_save_file_4 = '{}/glint-face/IJB/result/emore0.8/{}_result/cosface.npy'.format(
    root, target)
score_save_file_5 = '{}/glint-face/IJB/result/emore1.0/{}_result/cosface.npy'.format(
    root, target)
score_save_file_6 = '{}/glint-face/IJB/result/retina1.0/{}_result/cosface.npy'.format(
    root, target)

save_path = '{}/glint-face/IJB'.format(root)
image_path = '{}/face/IJB_release/{}'.format(root, target)
methods = [
    'celeb360k_final0.1, S=0.1', 'celeb360k_final1.0, S=1.0', 'MS1MV2, S=0.4',
    'MS1MV2, S=0.8', 'MS1MV2, S=1.0', 'RETINA, S=1.0'
]
files = [
    score_save_file_1, score_save_file_2, score_save_file_3, score_save_file_4,
    score_save_file_5, score_save_file_6
]


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % target.lower()))

scores = []
for file in files:
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
    plt.plot(
        fpr,
        tpr,
        color=colours[method],
        lw=1,
        # label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc * 100))
        label=method)
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        # tpr_fpr_row.append('%.4f' % tpr[min_index])
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10**-6, 0.1])
plt.ylim([0.30, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on {}'.format(title))
plt.legend(loc="lower right")
# plt.show()
fig.savefig(os.path.join(save_path, '%s.pdf' % job))
print(tpr_fpr_table)
