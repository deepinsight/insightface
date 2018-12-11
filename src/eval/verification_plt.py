"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from matplotlib import pyplot as plt

if __name__ == '__main__':
    tpr, fpr = pickle.load(open("tpr_fpr.bin", 'rb'))
    print(tpr)
    print(fpr)

    plt.rcParams['figure.figsize'] = (10, 5)
    plt.plot
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(fpr, tpr, 'r.', label="roc")
    ax1.set_xlabel("false presitive rate")
    ax1.set_ylabel("true presitive rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 1)
    ax1.legend(loc=4)

    val, far = pickle.load(open("val_far.bin", 'rb'))
    print(val)
    print(far)

    ax2 = fig.add_subplot(122)
    ax2.plot(fpr, tpr, 'r.', label="tar_far")
    ax2.set_xlabel("far")
    ax2.set_ylabel("tar")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, 1)
    ax2.legend(loc=4)
    plt.show()
