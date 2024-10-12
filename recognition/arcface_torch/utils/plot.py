import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc

with open(sys.argv[1], "r") as f:
    files = f.readlines()

files = [x.strip() for x in files]
image_path = "/train_tmp/IJB_release/IJBC"


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int32)
    t2 = pairs[:, 1].astype(np.int32)
    label = pairs[:, 2].astype(np.int32)
    return t1, t2, label


p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % 'ijbc'))

methods = []
scores = []
for file in files:
    methods.append(file)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
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
    tpr_fpr_row.append(method)
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
print(tpr_fpr_table)
