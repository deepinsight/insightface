from __future__ import division
import datetime
import os
import os.path as osp
import glob
import numpy as np
import cv2
import sys
import onnxruntime
import onnx
import argparse
from onnx import numpy_helper
from insightface.data import get_image

class ArcFaceORT:
    def __init__(self, model_path, cpu=False):
        self.model_path = model_path
        self.cpu = cpu

    #input_size is (w,h), return error message, return None if success
    def check(self, track='cfat', test_img = None):
        #default is cfat
        max_model_size_mb=1024
        max_feat_dim=512
        max_time_cost=15
        if track.startswith('ms1m'):
            max_model_size_mb=1024
            max_feat_dim=512
            max_time_cost=10
        elif track.startswith('glint'):
            max_model_size_mb=1024
            max_feat_dim=1024
            max_time_cost=20
        elif track.startswith('cfat'):
            max_model_size_mb = 1024
            max_feat_dim = 512
            max_time_cost = 15
        elif track.startswith('unconstrained'):
            max_model_size_mb=1024
            max_feat_dim=1024
            max_time_cost=30
        else:
            return "track not found"

        if not os.path.exists(self.model_path):
            return "model_path not exists"
        if not os.path.isdir(self.model_path):
            return "model_path should be directory"
        onnx_files = []
        for _file in os.listdir(self.model_path):
            if _file.endswith('.onnx'):
                onnx_files.append(osp.join(self.model_path, _file))
        if len(onnx_files)==0:
            return "do not have onnx files"
        self.model_file = sorted(onnx_files)[-1]
        print('use onnx-model:', self.model_file)
        try:
            session = onnxruntime.InferenceSession(self.model_file, None)
        except:
            return "load onnx failed"
        if self.cpu:
            session.set_providers(['CPUExecutionProvider'])
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        print('input-shape:', input_shape)
        if len(input_shape)!=4:
            return "length of input_shape should be 4"
        if not isinstance(input_shape[0], str):
            #return "input_shape[0] should be str to support batch-inference"
            print('reset input-shape[0] to None')
            model = onnx.load(self.model_file)
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
            new_model_file = osp.join(self.model_path, 'zzzzrefined.onnx')
            onnx.save(model, new_model_file)
            self.model_file = new_model_file
            print('use new onnx-model:', self.model_file)
            try:
                session = onnxruntime.InferenceSession(self.model_file, None)
            except:
                return "load onnx failed"
            if self.cpu:
                session.set_providers(['CPUExecutionProvider'])
            input_cfg = session.get_inputs()[0]
            input_shape = input_cfg.shape
            print('new-input-shape:', input_shape)

        self.image_size = tuple(input_shape[2:4][::-1])
        #print('image_size:', self.image_size)
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
            #print(o.name, o.shape)
        if len(output_names)!=1:
            return "number of output nodes should be 1"
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        #print(self.output_names)
        model = onnx.load(self.model_file)
        graph = model.graph
        if len(graph.node)<8:
            return "too small onnx graph"

        input_size = (112,112)
        self.crop = None
        if track=='cfat':
            crop_file = osp.join(self.model_path, 'crop.txt')
            if osp.exists(crop_file):
                lines = open(crop_file,'r').readlines()
                if len(lines)!=6:
                    return "crop.txt should contain 6 lines"
                lines = [int(x) for x in lines]
                self.crop = lines[:4]
                input_size = tuple(lines[4:6])
        if input_size!=self.image_size:
            return "input-size is inconsistant with onnx model input, %s vs %s"%(input_size, self.image_size)

        self.model_size_mb = os.path.getsize(self.model_file) / float(1024*1024)
        if self.model_size_mb > max_model_size_mb:
            return "max model size exceed, given %.3f-MB"%self.model_size_mb

        input_mean = None
        input_std = None
        if track=='cfat':
            pn_file = osp.join(self.model_path, 'pixel_norm.txt')
            if osp.exists(pn_file):
                lines = open(pn_file,'r').readlines()
                if len(lines)!=2:
                    return "pixel_norm.txt should contain 2 lines"
                input_mean = float(lines[0])
                input_std = float(lines[1])
        if input_mean is not None or input_std is not None:
            if input_mean is None or input_std is None:
                return "please set input_mean and input_std simultaneously"
        else:
            find_sub = False
            find_mul = False
            for nid, node in enumerate(graph.node[:8]):
                print(nid, node.name)
                if node.name.startswith('Sub') or node.name.startswith('_minus'):
                    find_sub = True
                if node.name.startswith('Mul') or node.name.startswith('_mul') or node.name.startswith('Div'):
                    find_mul = True
            if find_sub and find_mul:
                print("find sub and mul")
                #mxnet arcface model
                input_mean = 0.0
                input_std = 1.0
            else:
                input_mean = 127.5
                input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        for initn in graph.initializer:
            weight_array = numpy_helper.to_array(initn)
            dt = weight_array.dtype
            if dt.itemsize<4:
                return 'invalid weight type - (%s:%s)' % (initn.name, dt.name)
        if test_img is None:
            test_img = get_image('Tom_Hanks_54745')
            test_img = cv2.resize(test_img, self.image_size)
        else:
            test_img = cv2.resize(test_img, self.image_size)
        feat, cost = self.benchmark(test_img)
        batch_result = self.check_batch(test_img)
        batch_result_sum = float(np.sum(batch_result))
        if batch_result_sum in [float('inf'), -float('inf')] or batch_result_sum != batch_result_sum:
            print(batch_result)
            print(batch_result_sum)
            return "batch result output contains NaN!"

        if len(feat.shape) < 2:
           return "the shape of the feature must be two, but get {}".format(str(feat.shape))

        if feat.shape[1] > max_feat_dim:
            return "max feat dim exceed, given %d"%feat.shape[1]
        self.feat_dim = feat.shape[1]
        cost_ms = cost*1000
        if cost_ms>max_time_cost:
            return "max time cost exceed, given %.4f"%cost_ms
        self.cost_ms = cost_ms
        print('check stat:, model-size-mb: %.4f, feat-dim: %d, time-cost-ms: %.4f, input-mean: %.3f, input-std: %.3f'%(self.model_size_mb, self.feat_dim, self.cost_ms, self.input_mean, self.input_std))
        return None

    def check_batch(self, img):
        if not isinstance(img, list):
            imgs = [img, ] * 32
        if self.crop is not None:
            nimgs = []
            for img in imgs:
                nimg = img[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2], :]
                if nimg.shape[0] != self.image_size[1] or nimg.shape[1] != self.image_size[0]:
                    nimg = cv2.resize(nimg, self.image_size)
                nimgs.append(nimg)
            imgs = nimgs
        blob = cv2.dnn.blobFromImages(
            images=imgs, scalefactor=1.0 / self.input_std, size=self.image_size,
            mean=(self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out


    def meta_info(self):
        return {'model-size-mb':self.model_size_mb, 'feature-dim':self.feat_dim, 'infer': self.cost_ms}


    def forward(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.image_size
        if self.crop is not None:
            nimgs = []
            for img in imgs:
                nimg = img[self.crop[1]:self.crop[3],self.crop[0]:self.crop[2],:]
                if nimg.shape[0]!=input_size[1] or nimg.shape[1]!=input_size[0]:
                    nimg = cv2.resize(nimg, input_size)
                nimgs.append(nimg)
            imgs = nimgs
        blob = cv2.dnn.blobFromImages(imgs, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name : blob})[0]
        return net_out

    def benchmark(self, img):
        input_size = self.image_size
        if self.crop is not None:
            nimg = img[self.crop[1]:self.crop[3],self.crop[0]:self.crop[2],:]
            if nimg.shape[0]!=input_size[1] or nimg.shape[1]!=input_size[0]:
                nimg = cv2.resize(nimg, input_size)
            img = nimg
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        costs = []
        for _ in range(50):
            ta = datetime.datetime.now()
            net_out = self.session.run(self.output_names, {self.input_name : blob})[0]
            tb = datetime.datetime.now()
            cost = (tb-ta).total_seconds()
            costs.append(cost)
        costs = sorted(costs)
        cost = costs[5]
        return net_out, cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # general
    parser.add_argument('workdir', help='submitted work dir', type=str)
    parser.add_argument('--track', help='track name, for different challenge', type=str, default='cfat')
    args = parser.parse_args()
    handler = ArcFaceORT(args.workdir)
    err = handler.check(args.track)
    print('err:', err)
