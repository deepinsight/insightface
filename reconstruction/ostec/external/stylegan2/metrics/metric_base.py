# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Common definitions for GAN metrics."""

import os
import time
import hashlib
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc
from training import dataset

#----------------------------------------------------------------------------
# Base class for metrics.

class MetricBase:
    def __init__(self, name):
        self.name = name
        self._dataset_obj = None
        self._progress_lo = None
        self._progress_hi = None
        self._progress_max = None
        self._progress_sec = None
        self._progress_time = None
        self._reset()

    def close(self):
        self._reset()

    def _reset(self, network_pkl=None, run_dir=None, data_dir=None, dataset_args=None, mirror_augment=None):
        if self._dataset_obj is not None:
            self._dataset_obj.close()

        self._network_pkl = network_pkl
        self._data_dir = data_dir
        self._dataset_args = dataset_args
        self._dataset_obj = None
        self._mirror_augment = mirror_augment
        self._eval_time = 0
        self._results = []

        if (dataset_args is None or mirror_augment is None) and run_dir is not None:
            run_config = misc.parse_config_for_previous_run(run_dir)
            self._dataset_args = dict(run_config['dataset'])
            self._dataset_args['shuffle_mb'] = 0
            self._mirror_augment = run_config['train'].get('mirror_augment', False)

    def configure_progress_reports(self, plo, phi, pmax, psec=15):
        self._progress_lo = plo
        self._progress_hi = phi
        self._progress_max = pmax
        self._progress_sec = psec

    def run(self, network_pkl, run_dir=None, data_dir=None, dataset_args=None, mirror_augment=None, num_gpus=1, tf_config=None, log_results=True, Gs_kwargs=dict(is_validation=True)):
        self._reset(network_pkl=network_pkl, run_dir=run_dir, data_dir=data_dir, dataset_args=dataset_args, mirror_augment=mirror_augment)
        time_begin = time.time()
        with tf.Graph().as_default(), tflib.create_session(tf_config).as_default(): # pylint: disable=not-context-manager
            self._report_progress(0, 1)
            _G, _D, Gs = misc.load_pkl(self._network_pkl)
            self._evaluate(Gs, Gs_kwargs=Gs_kwargs, num_gpus=num_gpus)
            self._report_progress(1, 1)
        self._eval_time = time.time() - time_begin # pylint: disable=attribute-defined-outside-init

        if log_results:
            if run_dir is not None:
                log_file = os.path.join(run_dir, 'metric-%s.txt' % self.name)
                with dnnlib.util.Logger(log_file, 'a'):
                    print(self.get_result_str().strip())
            else:
                print(self.get_result_str().strip())

    def get_result_str(self):
        network_name = os.path.splitext(os.path.basename(self._network_pkl))[0]
        if len(network_name) > 29:
            network_name = '...' + network_name[-26:]
        result_str = '%-30s' % network_name
        result_str += ' time %-12s' % dnnlib.util.format_time(self._eval_time)
        for res in self._results:
            result_str += ' ' + self.name + res.suffix + ' '
            result_str += res.fmt % res.value
        return result_str

    def update_autosummaries(self):
        for res in self._results:
            tflib.autosummary.autosummary('Metrics/' + self.name + res.suffix, res.value)

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        raise NotImplementedError # to be overridden by subclasses

    def _report_result(self, value, suffix='', fmt='%-10.4f'):
        self._results += [dnnlib.EasyDict(value=value, suffix=suffix, fmt=fmt)]

    def _report_progress(self, pcur, pmax, status_str=''):
        if self._progress_lo is None or self._progress_hi is None or self._progress_max is None:
            return
        t = time.time()
        if self._progress_sec is not None and self._progress_time is not None and t < self._progress_time + self._progress_sec:
            return
        self._progress_time = t
        val = self._progress_lo + (pcur / pmax) * (self._progress_hi - self._progress_lo)
        dnnlib.RunContext.get().update(status_str, int(val), self._progress_max)

    def _get_cache_file_for_reals(self, extension='pkl', **kwargs):
        all_args = dnnlib.EasyDict(metric_name=self.name, mirror_augment=self._mirror_augment)
        all_args.update(self._dataset_args)
        all_args.update(kwargs)
        md5 = hashlib.md5(repr(sorted(all_args.items())).encode('utf-8'))
        dataset_name = self._dataset_args.get('tfrecord_dir', None) or self._dataset_args.get('h5_file', None)
        dataset_name = os.path.splitext(os.path.basename(dataset_name))[0]
        return os.path.join('.stylegan2-cache', '%s-%s-%s.%s' % (md5.hexdigest(), self.name, dataset_name, extension))

    def _get_dataset_obj(self):
        if self._dataset_obj is None:
            self._dataset_obj = dataset.load_dataset(data_dir=self._data_dir, **self._dataset_args)
        return self._dataset_obj

    def _iterate_reals(self, minibatch_size):
        dataset_obj = self._get_dataset_obj()
        while True:
            images, _labels = dataset_obj.get_minibatch_np(minibatch_size)
            if self._mirror_augment:
                images = misc.apply_mirror_augment(images)
            yield images

    def _iterate_fakes(self, Gs, minibatch_size, num_gpus):
        while True:
            latents = np.random.randn(minibatch_size, *Gs.input_shape[1:])
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, output_transform=fmt, is_validation=True, num_gpus=num_gpus, assume_frozen=True)
            yield images

    def _get_random_labels_tf(self, minibatch_size):
        return self._get_dataset_obj().get_random_labels_tf(minibatch_size)

#----------------------------------------------------------------------------
# Group of multiple metrics.

class MetricGroup:
    def __init__(self, metric_kwarg_list):
        self.metrics = [dnnlib.util.call_func_by_name(**kwargs) for kwargs in metric_kwarg_list]

    def run(self, *args, **kwargs):
        for metric in self.metrics:
            metric.run(*args, **kwargs)

    def get_result_str(self):
        return ' '.join(metric.get_result_str() for metric in self.metrics)

    def update_autosummaries(self):
        for metric in self.metrics:
            metric.update_autosummaries()

#----------------------------------------------------------------------------
# Dummy metric for debugging purposes.

class DummyMetric(MetricBase):
    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        _ = Gs, Gs_kwargs, num_gpus
        self._report_result(0.0)

#----------------------------------------------------------------------------
