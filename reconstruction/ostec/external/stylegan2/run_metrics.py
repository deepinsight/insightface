# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import os
import sys

import dnnlib
import dnnlib.tflib as tflib

import pretrained_networks
from metrics import metric_base
from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------

def run(network_pkl, metrics, dataset, data_dir, mirror_augment):
    print('Evaluating metrics "%s" for "%s"...' % (','.join(metrics), network_pkl))
    tflib.init_tf()
    network_pkl = pretrained_networks.get_path_or_url(network_pkl)
    dataset_args = dnnlib.EasyDict(tfrecord_dir=dataset, shuffle_mb=0)
    num_gpus = dnnlib.submit_config.num_gpus
    metric_group = metric_base.MetricGroup([metric_defaults[metric] for metric in metrics])
    metric_group.run(network_pkl, data_dir=data_dir, dataset_args=dataset_args, mirror_augment=mirror_augment, num_gpus=num_gpus)

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#----------------------------------------------------------------------------

_examples = '''examples:

  python %(prog)s --data-dir=~/datasets --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --metrics=fid50k,ppl_wend --dataset=ffhq --mirror-augment=true

valid metrics:

  ''' + ', '.join(sorted([x for x in metric_defaults.keys()])) + '''
'''

def main():
    parser = argparse.ArgumentParser(
        description='Run StyleGAN2 metrics.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--metrics', help='Metrics to compute (default: %(default)s)', default='fid50k', type=lambda x: x.split(','))
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=False, type=_str_to_bool, metavar='BOOL')
    parser.add_argument('--num-gpus', help='Number of GPUs to use', type=int, default=1, metavar='N')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = kwargs.pop('num_gpus')
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = 'run-metrics'
    dnnlib.submit_run(sc, 'run_metrics.run', **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
