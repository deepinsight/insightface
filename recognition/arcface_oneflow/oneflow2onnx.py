import os
from os import mkdir
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check
import oneflow as flow

import logging
from backbones import get_model
from utils.utils_config import get_config
import argparse
import tempfile


class ModelGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

    def build(self, x):
        x = x.to("cuda")
        out = self.backbone(x)
        return out


def convert_func(cfg, model_path, out_path,image_size):

    model_module = get_model(cfg.network, dropout=0.0,
                             num_features=cfg.embedding_size).to("cuda")
    model_module.eval()
    print(model_module)
    model_graph = ModelGraph(model_module)
    model_graph._compile(flow.randn(1, 3, image_size, image_size).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        new_parameters = dict()
        parameters = flow.load(model_path)
        for key, value in parameters.items():
            if "num_batches_tracked" not in key:
                if key == "fc.weight":
                    continue
                val = value
                new_key = key.replace("backbone.", "")
                new_parameters[new_key] = val
        model_module.load_state_dict(new_parameters)
        flow.save(model_module.state_dict(), tmpdirname)
        convert_to_onnx_and_check(
            model_graph, flow_weight_dir=tmpdirname, onnx_model_path="./", print_outlier=True)


def main(args):
    logging.basicConfig(level=logging.NOTSET)
    logging.info(args.model_path)
    cfg = get_config(args.config)
    if not os.path.exists(args.out_path):
        mkdir(args.out_path)
    convert_func(cfg, args.model_path, args.out_path,args.image_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OneFlow ArcFace val')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--image_size', type=int,
                        default=112, help='input image size')
    parser.add_argument('--out_path', type=str,
                        default="onnx_model", help='out path')

