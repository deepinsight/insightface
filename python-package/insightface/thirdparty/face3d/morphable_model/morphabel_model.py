from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
from .. import mesh
from . import fit
from . import load

class  MorphabelModel(object):
    """docstring for  MorphabelModel
    model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
            'shapeMU': [3*nver, 1]. *
            'shapePC': [3*nver, n_shape_para]. *
            'shapeEV': [n_shape_para, 1]. ~
            'expMU': [3*nver, 1]. ~ 
            'expPC': [3*nver, n_exp_para]. ~
            'expEV': [n_exp_para, 1]. ~
            'texMU': [3*nver, 1]. ~
            'texPC': [3*nver, n_tex_para]. ~
            'texEV': [n_tex_para, 1]. ~
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
            'kpt_ind': [68,] (start from 1). ~
    """
    def __init__(self, model_path, model_type = 'BFM'):
        super( MorphabelModel, self).__init__()
        if model_type=='BFM':
            self.model = load.load_BFM(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()
            
        # fixed attributes
        self.nver = self.model['shapePC'].shape[0]/3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texMU'].shape[1]
        
        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']
        self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))

    # ------------------------------------- shape: represented with mesh(vertices & triangles(fixed))
    def get_shape_para(self, type = 'random'):
        if type == 'zero':
            sp = np.random.zeros((self.n_shape_para, 1))
        elif type == 'random':
            sp = np.random.rand(self.n_shape_para, 1)*1e04
        return sp

    def get_exp_para(self, type = 'random'):
        if type == 'zero':
            ep = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            ep = -1.5 + 3*np.random.random([self.n_exp_para, 1])
            ep[6:, 0] = 0

        return ep 

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1) 
        Returns:
            vertices: (nver, 3)
        '''
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape_para) + self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T

        return vertices

    # -------------------------------------- texture: here represented with rgb value(colors) in vertices.
    def get_tex_para(self, type = 'random'):
        if type == 'zero':
            tp = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            tp = np.random.rand(self.n_tex_para, 1)
        return tp

    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''
        colors = self.model['texMU'] + self.model['texPC'].dot(tex_para*self.model['texEV'])
        colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F').T/255.  
        
        return colors


    # ------------------------------------------- transformation
    # -------------  transform
    def rotate(self, vertices, angles):
        ''' rotate face
        Args:
            vertices: [nver, 3]
            angles: [3] x, y, z rotation angle(degree)
            x: pitch. positive for looking down 
            y: yaw. positive for looking left
            z: roll. positive for tilting head right
        Returns:
            vertices: rotated vertices
        '''
        return mesh.transform.rotate(vertices, angles)

    def transform(self, vertices, s, angles, t3d):
        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    def transform_3ddfa(self, vertices, s, angles, t3d): # only used for processing 300W_LP data
        R = mesh.transform.angle2matrix_3ddfa(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    # --------------------------------------------------- fitting
    def fit(self, x, X_ind, max_iter = 4, isShow = False):
        ''' fit 3dmm & pose parameters
        Args:
            x: (n, 2) image points
            X_ind: (n,) corresponding Model vertex indices
            max_iter: iteration
            isShow: whether to reserve middle results for show
        Returns:
            fitted_sp: (n_sp, 1). shape parameters
            fitted_ep: (n_ep, 1). exp parameters
            s, angles, t
        '''
        if isShow:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points_for_show(x, X_ind, self.model, n_sp = self.n_shape_para, n_ep = self.n_exp_para, max_iter = max_iter)
            angles = np.zeros((R.shape[0], 3))
            for i in range(R.shape[0]):
                angles[i] = mesh.transform.matrix2angle(R[i])
        else:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points(x, X_ind, self.model, n_sp = self.n_shape_para, n_ep = self.n_exp_para, max_iter = max_iter)
            angles = mesh.transform.matrix2angle(R)
        return fitted_sp, fitted_ep, s, angles, t


