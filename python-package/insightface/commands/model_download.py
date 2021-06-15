from argparse import ArgumentParser

from . import BaseInsightFaceCLICommand
import os
import os.path as osp
import zipfile
import glob
from ..utils import download, check_sha1


def model_download_command_factory(args):
    return ModelDownloadCommand(args.model, args.root, args.force)


class ModelDownloadCommand(BaseInsightFaceCLICommand):
    #_url_format = '{repo_url}models/{file_name}.zip'
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("model.download")
        download_parser.add_argument(
            "--root", type=str, default='~/.insightface', help="Path to location to store the models"
        )
        download_parser.add_argument(
            "--force", action="store_true", help="Force the model to be download even if already in root-dir"
        )
        download_parser.add_argument("model", type=str, help="Name of the model to download")
        download_parser.set_defaults(func=model_download_command_factory)

    def __init__(self, model: str, root: str, force: bool):
        self.base_repo_url = 'http://storage.insightface.ai/files/'
        self._model = model
        self._root = os.path.expanduser(root)
        self._force = force

    def run(self):
        if not os.path.exists(self._root):
            os.makedirs(self._root)
        dir_path = os.path.join(self._root, 'models', self._model)
        if osp.exists(dir_path) and not self._force:
            return
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        print('dir_path:', dir_path)
        zip_file_path = os.path.join(self._root, 'models', self._model + '.zip')
        model_url = "%s/models/%s.zip"%(self.base_repo_url, self._model)
        download(model_url,
                 path=zip_file_path,
                 overwrite=True)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(dir_path)
        os.remove(zip_file_path)

