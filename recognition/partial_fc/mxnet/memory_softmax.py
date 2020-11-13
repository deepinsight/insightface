import mxnet as mx
from mxnet import autograd
from mxnet import nd


class MarginLoss(object):
    """ Default is Arcface loss
    """
    def __init__(self, margins=(1.0, 0.5, 0.0), loss_s=64, embedding_size=512):
        """
        """
        # margins
        self.loss_m1 = margins[0]
        self.loss_m2 = margins[1]
        self.loss_m3 = margins[2]
        self.loss_s = loss_s
        self.embedding_size = embedding_size

    def forward(self, data, weight, mapping_label, depth):
        """
        """
        with autograd.record():
            norm_data = nd.L2Normalization(data)
            norm_weight = nd.L2Normalization(weight)
            #
            fc7 = nd.dot(norm_data, norm_weight, transpose_b=True)
            #
            mapping_label_onehot = mx.nd.one_hot(indices=mapping_label,
                                                 depth=depth,
                                                 on_value=1.0,
                                                 off_value=0.0)
            # cosface
            if self.loss_m1 == 1.0 and self.loss_m2 == 0.0:
                _one_hot = mapping_label_onehot * self.loss_m3
                fc7 = fc7 - _one_hot
            else:
                fc7_onehot = fc7 * mapping_label_onehot
                cos_t = fc7_onehot
                t = nd.arccos(cos_t)
                if self.loss_m1 != 1.0:
                    t = t * self.loss_m1
                if self.loss_m2 != 0.0:
                    t = t + self.loss_m2
                margin_cos = nd.cos(t)
                if self.loss_m3 != 0.0:
                    margin_cos = margin_cos - self.loss_m3
                margin_fc7 = margin_cos
                margin_fc7_onehot = margin_fc7 * mapping_label_onehot
                diff = margin_fc7_onehot - fc7_onehot
                fc7 = fc7 + diff

            fc7 = fc7 * self.loss_s
            return fc7, mapping_label_onehot
