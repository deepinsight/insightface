from argparse import ArgumentParser

from . import BaseInsightFaceCLICommand
import os
import os.path as osp
import zipfile
import glob
from ..utils import download


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
        self._model = model
        self._root = root
        self._force = force

    def run(self):
        download('models', self._model, force=self._force, root=self._root)

