
import os
import os.path as osp
import zipfile
from .download import download_file

BASE_REPO_URL='http://storage.insightface.ai/files'

def download(sub_dir, name, force=False, root='~/.insightface'):
    _root = os.path.expanduser(root)
    dir_path = os.path.join(_root, sub_dir, name)
    if osp.exists(dir_path) and not force:
        return dir_path
    print('download_path:', dir_path)
    zip_file_path = os.path.join(_root, sub_dir, name + '.zip')
    model_url = "%s/%s/%s.zip"%(BASE_REPO_URL, sub_dir, name)
    download_file(model_url,
             path=zip_file_path,
             overwrite=True)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(dir_path)
    os.remove(zip_file_path)
    return dir_path

def ensure_available(sub_dir, name, root='~/.insightface'):
    return download(sub_dir, name, force=False, root=root)

