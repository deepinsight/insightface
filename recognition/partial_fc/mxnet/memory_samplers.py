import numpy as np


class CenterPositiveClassGet(object):
    """ Get the corresponding center of the positive class
    """
    def __init__(self, num_sample, num_local, rank):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.rank_class_start = self.rank * num_local
        self.rank_class_end = self.rank_class_start + num_local
        pass

    def __call__(self, global_label):
        """
        Return:
        -------
        positive_center_label: list of int
        """
        greater_than = global_label >= self.rank_class_start
        smaller_than = global_label < self.rank_class_end

        positive_index = greater_than * smaller_than
        positive_center_label = global_label[positive_index]

        return positive_center_label


class CenterNegetiveClassSample(object):
    """ Sample negative class center
    """
    def __init__(self, num_sample, num_local, rank):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.negative_class_pool = np.arange(num_local)
        pass

    def __call__(self, positive_center_index):
        """
        Return:
        -------
        negative_center_index: list of int
        """
        negative_class_pool = np.setdiff1d(self.negative_class_pool,
                                           positive_center_index)
        negative_sample_size = self.num_sample - len(positive_center_index)
        negative_center_index = np.random.choice(negative_class_pool,
                                                 negative_sample_size,
                                                 replace=False)
        return negative_center_index


class WeightIndexSampler(object):
    def __init__(self, num_sample, num_local, rank):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.rank_class_start = self.rank * num_local

        self.positive = CenterPositiveClassGet(num_sample, num_local, rank)
        self.negative = CenterNegetiveClassSample(num_sample, num_local, rank)

    def __call__(self, global_label):
        positive_center_label = self.positive(global_label)
        positive_center_index = positive_center_label - self.positive.rank_class_start
        if len(positive_center_index) > self.num_sample:
            positive_center_index = positive_center_index[:self.num_sample]
        negative_center_index = self.negative(positive_center_index)
        #
        final_center_index = np.concatenate(
            (positive_center_index, negative_center_index))
        assert len(final_center_index) == len(
            np.unique(final_center_index)) == self.num_sample
        assert len(final_center_index) == self.num_sample
        return final_center_index
