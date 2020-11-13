from __future__ import division
import collections
import mxnet as mx
import numpy as np
from numpy.linalg import norm
import mxnet.ndarray as nd
from ..model_zoo import model_zoo
from ..utils import face_align

__all__ = ['FaceAnalysis', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class FaceAnalysis:
    def __init__(self,
                 det_name='retinaface_r50_v1',
                 rec_name='arcface_r100_v1',
                 ga_name='genderage_v1'):
        assert det_name is not None
        self.det_model = model_zoo.get_model(det_name)
        if rec_name is not None:
            self.rec_model = model_zoo.get_model(rec_name)
        else:
            self.rec_model = None
        if ga_name is not None:
            self.ga_model = model_zoo.get_model(ga_name)
        else:
            self.ga_model = None

    def prepare(self, ctx_id, nms=0.4):
        self.det_model.prepare(ctx_id, nms)
        if self.rec_model is not None:
            self.rec_model.prepare(ctx_id)
        if self.ga_model is not None:
            self.ga_model.prepare(ctx_id)

    def get(self, img, det_thresh=0.8, det_scale=1.0, max_num=0):
        bboxes, landmarks = self.det_model.detect(img,
                                                  threshold=det_thresh,
                                                  scale=det_scale)
        if bboxes.shape[0] == 0:
            return []
        if max_num > 0 and bboxes.shape[0] > max_num:
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] -
                                                    bboxes[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (bboxes[:, 0] + bboxes[:, 2]) / 2 - img_center[1],
                (bboxes[:, 1] + bboxes[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            bboxes = bboxes[bindex, :]
            landmarks = landmarks[bindex, :]
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            landmark = landmarks[i]
            _img = face_align.norm_crop(img, landmark=landmark)
            embedding = None
            embedding_norm = None
            normed_embedding = None
            gender = None
            age = None
            if self.rec_model is not None:
                embedding = self.rec_model.get_embedding(_img).flatten()
                embedding_norm = norm(embedding)
                normed_embedding = embedding / embedding_norm
            if self.ga_model is not None:
                gender, age = self.ga_model.get(_img)
            face = Face(bbox=bbox,
                        landmark=landmark,
                        det_score=det_score,
                        embedding=embedding,
                        gender=gender,
                        age=age,
                        normed_embedding=normed_embedding,
                        embedding_norm=embedding_norm)
            ret.append(face)
        return ret
