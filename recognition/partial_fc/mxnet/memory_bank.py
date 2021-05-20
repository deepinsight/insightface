import os

import numpy as np
from mxnet import nd
import mxnet as mx

from memory_samplers import WeightIndexSampler


class MemoryBank(object):
    def __init__(self,
                 num_sample,
                 num_local,
                 rank,
                 local_rank,
                 embedding_size,
                 prefix,
                 gpu=True):
        """
        Parameters
        ----------
        num_sample: int
            The number of sampled class center.
        num_local: int
            The number of class center storage in this rank(CPU/GPU).
        rank: int
            Unique process(GPU) ID from 0 to size - 1.
        local_rank: int
            Unique process(GPU) ID within the server from 0 to 7.
        embedding_size: int
            The feature dimension.
        prefix_dir: str
            Path prefix of model dir.
        gpu: bool
            If True, class center and class center mom will storage in GPU.
        """
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.embedding_size = embedding_size
        self.gpu = gpu
        self.prefix = prefix

        if self.gpu:
            context = mx.gpu(local_rank)
        else:
            context = mx.cpu()

        # In order to apply update, weight and momentum should be storage.
        self.weight = nd.random_normal(loc=0,
                                       scale=0.01,
                                       shape=(self.num_local,
                                              self.embedding_size),
                                       ctx=context)
        self.weight_mom = nd.zeros_like(self.weight)

        # Sampler object
        self.weight_index_sampler = WeightIndexSampler(num_sample, num_local,
                                                       rank)

    def sample(self, global_label):
        """
        Parameters
        ----------
        global_label: NDArray
            Global label (after gathers label from all rank)
        Returns
        -------
        index: ndarray(numpy)
            Local index for memory bank to sample, start from 0 to num_local, length is num_sample.
        global_label: ndarray(numpy)
            Global label after sort and unique.
        """
        assert isinstance(global_label, nd.NDArray)
        global_label = global_label.asnumpy()
        global_label = np.unique(global_label)
        global_label.sort()
        index = self.weight_index_sampler(global_label)
        index.sort()
        return index, global_label

    def get(self, index):
        """
        Get sampled class centers and their momentum.

        Parameters
        ----------
        index: NDArray
            Local index for memory bank to sample, start from 0 to num_local.
        """
        return self.weight[index], self.weight_mom[index]

    def set(self, index, updated_weight, updated_weight_mom=None):
        """
        Update sampled class to memory bank, make the class center stored
        in the memory bank the latest.

        Parameters
        ----------
        index: NDArray
            Local index for memory bank to sample, start from 0 to num_local.
        updated_weight: NDArray
            Class center which has been applied gradients.
        updated_weight_mom: NDArray
            Class center momentum which has been moved average.
        """

        self.weight[index] = updated_weight
        self.weight_mom[index] = updated_weight_mom

    def save(self):
        nd.save(fname=os.path.join(self.prefix,
                                   "%d_centers.param" % self.rank),
                data=self.weight)
        nd.save(fname=os.path.join(self.prefix,
                                   "%d_centers_mom.param" % self.rank),
                data=self.weight_mom)
