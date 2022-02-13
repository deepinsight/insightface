from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio

### ---------------------------------  load BFM data
def load_BFM(model_path):
    ''' load BFM 3DMM model
    Args:
        model_path: path to BFM model. 
    Returns:
        model: (nver = 53215, ntri = 105840). nver: number of vertices. ntri: number of triangles.
            'shapeMU': [3*nver, 1]
            'shapePC': [3*nver, 199]
            'shapeEV': [199, 1]
            'expMU': [3*nver, 1]
            'expPC': [3*nver, 29]
            'expEV': [29, 1]
            'texMU': [3*nver, 1]
            'texPC': [3*nver, 199]
            'texEV': [199, 1]
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles)
            'kpt_ind': [68,] (start from 1)
    PS:
        You can change codes according to your own saved data.
        Just make sure the model has corresponding attributes.
    '''
    C = sio.loadmat(model_path)
    model = C['model']
    model = model[0,0]

    # change dtype from double(np.float64) to np.float32, 
    # since big matrix process(espetially matrix dot) is too slow in python.
    model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
    model['shapePC'] = model['shapePC'].astype(np.float32)
    model['shapeEV'] = model['shapeEV'].astype(np.float32)
    model['expEV'] = model['expEV'].astype(np.float32)
    model['expPC'] = model['expPC'].astype(np.float32)

    # matlab start with 1. change to 0 in python.
    model['tri'] = model['tri'].T.copy(order = 'C').astype(np.int32) - 1
    model['tri_mouth'] = model['tri_mouth'].T.copy(order = 'C').astype(np.int32) - 1
    
    # kpt ind
    model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

    return model

def load_BFM_info(path = 'BFM_info.mat'):
    ''' load 3DMM model extra information
    Args:
        path: path to BFM info. 
    Returns:  
        model_info:
            'symlist': 2 x 26720
            'symlist_tri': 2 x 52937
            'segbin': 4 x n (0: nose, 1: eye, 2: mouth, 3: cheek)
            'segbin_tri': 4 x ntri 
            'face_contour': 1 x 28
            'face_contour_line': 1 x 512
            'face_contour_front': 1 x 28
            'face_contour_front_line': 1 x 512
            'nose_hole': 1 x 142
            'nose_hole_right': 1 x 71
            'nose_hole_left': 1 x 71
            'parallel': 17 x 1 cell
            'parallel_face_contour': 28 x 1 cell
            'uv_coords': n x 2
    '''
    C = sio.loadmat(path)
    model_info = C['model_info']
    model_info = model_info[0,0]
    return model_info

def load_uv_coords(path = 'BFM_UV.mat'):
    ''' load uv coords of BFM
    Args:
        path: path to data.
    Returns:  
        uv_coords: [nver, 2]. range: 0-1
    '''
    C = sio.loadmat(path)
    uv_coords = C['UV'].copy(order = 'C')
    return uv_coords

def load_pncc_code(path = 'pncc_code.mat'):
    ''' load pncc code of BFM
    PNCC code: Defined in 'Face Alignment Across Large Poses: A 3D Solution Xiangyu'
    download at http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm.
    Args:
        path: path to data.
    Returns:  
        pncc_code: [nver, 3]
    '''
    C = sio.loadmat(path)
    pncc_code = C['vertex_code'].T
    return pncc_code

## 
def get_organ_ind(model_info):
    ''' get nose, eye, mouth index
    '''
    valid_bin = model_info['segbin'].astype(bool)
    organ_ind = np.nonzero(valid_bin[0,:])[0]
    for i in range(1, valid_bin.shape[0] - 1):
        organ_ind = np.union1d(organ_ind, np.nonzero(valid_bin[i,:])[0])
    return organ_ind.astype(np.int32)
