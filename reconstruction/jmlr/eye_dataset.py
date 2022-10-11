import os
import os.path as osp
import numpy as np

import menpo.io as mio

def project_shape_in_image(verts, R_t, M_proj, M1):
    verts_homo = verts
    if verts_homo.shape[1] == 3:
        ones = np.ones([verts_homo.shape[0], 1])
        verts_homo = np.concatenate([verts_homo, ones], axis=1)
    verts_out = verts_homo @ R_t @ M_proj @ M1
    w_ = verts_out[:, [3]]
    verts_out = verts_out / w_
    return verts_out

class EyeDataset():
    def __init__(self, root, load_data=True):
        eyes_info = mio.import_pickle(osp.join(root,'eyes3d.pkl'))
        idxs481 = eyes_info['mask481']['idxs']
        tri481 = eyes_info['mask481']['trilist']
        self.iris_idx_481 = eyes_info['mask481']['idxs_iris']
        eyel_template = eyes_info['left_points'][idxs481]
        eyer_template = eyes_info['right_points'][idxs481]
        eyel_template_homo = np.append(eyel_template, np.ones((eyel_template.shape[0],1)), axis=1)
        eyer_template_homo = np.append(eyer_template, np.ones((eyer_template.shape[0],1)), axis=1)
        points = mio.import_pickle(osp.join(root,'eyespoints.pkl'))
        self.homol = eyel_template_homo.T
        self.homor = eyer_template_homo.T
        if load_data:
            self.worldl = {}
            self.worldr = {}
            #vector_norm = 0.035
            for k in points:
                p = k.find('/')
                newk = k[p+1:]
                value = points[k]
                #el_inv = (value['left'] @ eyel_template_homo.T).T
                #er_inv = (value['right'] @ eyer_template_homo.T).T
                #print('V:', value['left'][:5,:])
                #print('E:', el_inv[:5,:])
                # gaze vector of left eye in world space
                #gl_vector = el_inv[iris_idx_481].mean(axis=0) - el_inv[-1]
                #gl_vector = (gl_vector / np.linalg.norm(gl_vector)) * vector_norm
                #gl_point = el_inv[iris_idx_481].mean(axis=0) + gl_vector
                ## gaze vector of right eye in world space
                #gr_vector = er_inv[iris_idx_481].mean(axis=0) - er_inv[-1]
                #gr_vector = (gr_vector / np.linalg.norm(gr_vector)) * vector_norm
                #gr_point = er_inv[iris_idx_481].mean(axis=0) + gr_vector
                #self.world[newk] = (el_inv, er_inv, gl_point, gr_point)
                self.worldl[newk] = value['left']
                self.worldr[newk] = value['right']
            #print(self.points.keys())

    def get(self, key, to_homo=False):
        if key not in self.worldl:
            return None, None
        left = self.worldl[key]
        right = self.worldr[key]
        if to_homo:
            left = (left @ self.homol).T
            right = (right @ self.homor).T
        return left, right

    def to_homo(self, eyel, eyer):
        eyel = (eyel @ self.homol).T
        eyer = (eyer @ self.homor).T
        return eyel, eyer



