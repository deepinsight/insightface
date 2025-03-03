# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm
import cv2

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available
from .common import Face

__all__ = ['FaceAnalysis']


class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = ensure_available('models', name, root=root)
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    @staticmethod
    def clahe_preprocess_colour(img: np.array):
        """
        Function to perform CLAHE Histogram Equalization on image
        args:
        images (np.array): Numpy array representation of the image
        returns:
        output (np.array): Numpy array representation of the preprocessed image
        """

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(2, 2))

        # Equalize the histogram of the Y channel
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

        # Convert the YUV image back to RGB format
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def get(
            self,
            img,
            op_type,
            max_num=0,
            det_metric='default',
            preprocess_colour=None,
            dominant_face=None,
            min_perimeter=0
    ):

        if preprocess_colour and preprocess_colour == 'clahe':
            img = self.clahe_preprocess_colour(img)

        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric=det_metric)

        if bboxes.shape[0] == 0:
            return []

        # fr_trigger => left
        # onboarding => largest
        if dominant_face == 'left':
            # Initialize variables for finding dominant face
            dominant_ind, dominant_left = 10000, 10000

            # Process each detected face
            for i in range(len(bboxes)):
                left, top, right, bottom, score = bboxes[i]
                perimeter = (right - left) + (bottom - top)

                if left < dominant_left and perimeter > min_perimeter:
                    dominant_left = left
                    dominant_ind = i

            # Filter to keep only the dominant face
            bboxes = np.array([bboxes[dominant_ind]])
            kpss = np.array([kpss[dominant_ind]])

        elif dominant_face == 'largest':
            dominant_ind, dominant_per = 10000, -10000
            for i in range(len(bboxes)):
                left, top, right, bottom, score = bboxes[i]
                perimeter = (right - left) + (bottom - top)

                if perimeter > dominant_per:
                    dominant_per = perimeter
                    dominant_ind = i

            bboxes = np.array([bboxes[dominant_ind]])
            kpss = np.array([kpss[dominant_ind]])

        else:
            raise Exception(f"Invalid op_type! type= {op_type}")

        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                # print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg, '%s,%d' % (face.sex, face.age), (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 0), 1)

            # for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg