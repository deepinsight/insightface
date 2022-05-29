# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""List of pre-trained StyleGAN2 networks located on Google Drive."""

import pickle
import external.stylegan2.dnnlib
from external.stylegan2.dnnlib import tflib as tflib

#----------------------------------------------------------------------------
# StyleGAN2 Google Drive root: https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7

gdrive_urls = {
    'gdrive:networks/stylegan2-car-config-a.pkl':                           'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-a.pkl',
    'gdrive:networks/stylegan2-car-config-b.pkl':                           'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-b.pkl',
    'gdrive:networks/stylegan2-car-config-c.pkl':                           'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-c.pkl',
    'gdrive:networks/stylegan2-car-config-d.pkl':                           'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-d.pkl',
    'gdrive:networks/stylegan2-car-config-e.pkl':                           'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-e.pkl',
    'gdrive:networks/stylegan2-car-config-f.pkl':                           'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl',
    'gdrive:networks/stylegan2-cat-config-a.pkl':                           'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-a.pkl',
    'gdrive:networks/stylegan2-cat-config-f.pkl':                           'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-f.pkl',
    'gdrive:networks/stylegan2-church-config-a.pkl':                        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-a.pkl',
    'gdrive:networks/stylegan2-church-config-f.pkl':                        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl',
    'gdrive:networks/stylegan2-ffhq-config-a.pkl':                          'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-a.pkl',
    'gdrive:networks/stylegan2-ffhq-config-b.pkl':                          'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-b.pkl',
    'gdrive:networks/stylegan2-ffhq-config-c.pkl':                          'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-c.pkl',
    'gdrive:networks/stylegan2-ffhq-config-d.pkl':                          'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-d.pkl',
    'gdrive:networks/stylegan2-ffhq-config-e.pkl':                          'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-e.pkl',
    'gdrive:networks/stylegan2-ffhq-config-f.pkl':                          'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
    'gdrive:networks/stylegan2-horse-config-a.pkl':                         'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-horse-config-a.pkl',
    'gdrive:networks/stylegan2-horse-config-f.pkl':                         'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-horse-config-f.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl':    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl':   'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl',
}

#----------------------------------------------------------------------------

def get_path_or_url(path_or_gdrive_path):
    return gdrive_urls.get(path_or_gdrive_path, path_or_gdrive_path)

#----------------------------------------------------------------------------

_cached_networks = dict()

def load_networks(path_or_gdrive_path):
    path_or_url = get_path_or_url(path_or_gdrive_path)
    if path_or_url in _cached_networks:
        return _cached_networks[path_or_url]

    if external.stylegan2.dnnlib.util.is_url(path_or_url):
        stream = external.stylegan2.dnnlib.util.open_url(path_or_url, cache_dir='.stylegan2-cache')
    else:
        stream = open(path_or_url, 'rb')

    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    _cached_networks[path_or_url] = G, D, Gs
    return G, D, Gs

#----------------------------------------------------------------------------
