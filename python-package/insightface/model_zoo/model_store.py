"""
This code file mainly comes from https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/model_store.py
"""
from __future__ import print_function

__all__ = ['get_model_file']
import os
import zipfile
import glob

from ..utils import download, check_sha1

_model_sha1 = {
    name: checksum
    for checksum, name in [
        ('95be21b58e29e9c1237f229dae534bd854009ce0', 'arcface_r100_v1'),
        ('', 'arcface_mfn_v1'),
        ('39fd1e087a2a2ed70a154ac01fecaa86c315d01b', 'retinaface_r50_v1'),
        ('2c9de8116d1f448fd1d4661f90308faae34c990a', 'retinaface_mnet025_v1'),
        ('0db1d07921d005e6c9a5b38e059452fc5645e5a4', 'retinaface_mnet025_v2'),
        ('7dd8111652b7aac2490c5dcddeb268e53ac643e6', 'genderage_v1'),
    ]
}

base_repo_url = 'https://insightface.ai/files/'
_url_format = '{repo_url}models/{file_name}.zip'


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError(
            'Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def find_params_file(dir_path):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob("%s/*.params" % dir_path)
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]


def get_model_file(name, root=os.path.join('~', '.insightface', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """

    file_name = name
    root = os.path.expanduser(root)
    dir_path = os.path.join(root, name)
    file_path = find_params_file(dir_path)
    #file_path = os.path.join(root, file_name + '.params')
    sha1_hash = _model_sha1[name]
    if file_path is not None:
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print(
                'Mismatch in the content of model file detected. Downloading again.'
            )
    else:
        print('Model file is not found. Downloading.')

    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    zip_file_path = os.path.join(root, file_name + '.zip')
    repo_url = base_repo_url
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(dir_path)
    os.remove(zip_file_path)
    file_path = find_params_file(dir_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError(
            'Downloaded file has different hash. Please try again.')
