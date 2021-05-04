# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

import os
import os.path as osp
import glob
import onnxruntime
from .arcface_onnx import *
from .scrfd import *

#__all__ = ['get_model', 'get_model_list', 'get_arcface_onnx', 'get_scrfd']
__all__ = ['get_model']


class ModelRouter:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file

    def get_model(self):
        session = onnxruntime.InferenceSession(self.onnx_file, None)
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()
        #print(input_shape)
        if len(outputs)>=5:
            return SCRFD(model_file=self.onnx_file, session=session)
        elif input_shape[2]==112 and input_shape[3]==112:
            return ArcFaceONNX(model_file=self.onnx_file, session=session)
        else:
            raise RuntimeError('error on model routing')

def find_onnx_file(dir_path):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob("%s/*.onnx" % dir_path)
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]

def get_model(name, **kwargs):
    root = kwargs.get('root', '~/.insightface/models')
    root = os.path.expanduser(root)
    if not name.endswith('.onnx'):
        model_dir = os.path.join(root, name)
        model_file = find_onnx_file(model_dir)
        if model_file is None:
            return None
    else:
        model_file = name
    assert osp.isfile(model_file), 'model should be file'
    router = ModelRouter(name)
    model = router.get_model()
    #print('get-model for ', name,' : ', model.taskname)
    return model

