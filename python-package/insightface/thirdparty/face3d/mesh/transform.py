'''
Functions about transforming mesh(changing the position: modify vertices).
1. forward: transform(transform, camera, project).
2. backward: estimate transform matrix from correspondences.

Author: Yao Feng 
Mail: yaofeng1995@gmail.com
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
from math import cos, sin

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

def angle2matrix_3ddfa(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw. 
        z: roll. 
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]
    
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  sin(x)],
                 [0, -sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, -sin(y)],
                 [      0, 1,      0],
                 [sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), sin(z), 0],
                 [-sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)


## ------------------------------------------ 1. transform(transform, project, camera).
## ---------- 3d-3d transform. Transform obj in world space
def rotate(vertices, angles):
    ''' rotate vertices. 
    X_new = R.dot(X). X: 3 x 1   
    Args:
        vertices: [nver, 3]. 
        rx, ry, rz: degree angles
        rx: pitch. positive for looking down 
        ry: yaw. positive for looking left
        rz: roll. positive for tilting head right
    Returns:
        rotated vertices: [nver, 3]
    '''
    R = angle2matrix(angles)
    rotated_vertices = vertices.dot(R.T)

    return rotated_vertices

def similarity_transform(vertices, s, R, t3d):
    ''' similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    Args:(float32)
        vertices: [nver, 3]. 
        s: [1,]. scale factor.
        R: [3,3]. rotation matrix.
        t3d: [3,]. 3d translation vector.
    Returns:
        transformed vertices: [nver, 3]
    '''
    t3d = np.squeeze(np.array(t3d, dtype = np.float32))
    transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    return transformed_vertices


## -------------- Camera. from world space to camera space
# Ref: https://cs184.eecs.berkeley.edu/lecture/transforms-2
def normalize(x):
    epsilon = 1e-12
    norm = np.sqrt(np.sum(x**2, axis = 0))
    norm = np.maximum(norm, epsilon)
    return x/norm

def lookat_camera(vertices, eye, at = None, up = None):
    """ 'look at' transformation: from world space to camera space
    standard camera space: 
        camera located at the origin. 
        looking down negative z-axis. 
        vertical vector is y-axis.
    Xcam = R(X - C)
    Homo: [[R, -RC], [0, 1]]
    Args:
      vertices: [nver, 3] 
      eye: [3,] the XYZ world space position of the camera.
      at: [3,] a position along the center of the camera's gaze.
      up: [3,] up direction 
    Returns:
      transformed_vertices: [nver, 3]
    """
    if at is None:
      at = np.array([0, 0, 0], np.float32)
    if up is None:
      up = np.array([0, 1, 0], np.float32)

    eye = np.array(eye).astype(np.float32)
    at = np.array(at).astype(np.float32)
    z_aixs = -normalize(at - eye) # look forward
    x_aixs = normalize(np.cross(up, z_aixs)) # look right
    y_axis = np.cross(z_aixs, x_aixs) # look up

    R = np.stack((x_aixs, y_axis, z_aixs))#, axis = 0) # 3 x 3
    transformed_vertices = vertices - eye # translation
    transformed_vertices = transformed_vertices.dot(R.T) # rotation
    return transformed_vertices

## --------- 3d-2d project. from camera space to image plane
# generally, image plane only keeps x,y channels, here reserve z channel for calculating z-buffer.
def orthographic_project(vertices):
    ''' scaled orthographic projection(just delete z)
        assumes: variations in depth over the object is small relative to the mean distance from camera to object
        x -> x*f/z, y -> x*f/z, z -> f.
        for point i,j. zi~=zj. so just delete z
        ** often used in face
        Homo: P = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    Args:
        vertices: [nver, 3]
    Returns:
        projected_vertices: [nver, 3] if isKeepZ=True. [nver, 2] if isKeepZ=False.
    '''
    return vertices.copy()

def perspective_project(vertices, fovy, aspect_ratio = 1., near = 0.1, far = 1000.):
    ''' perspective projection.
    Args:
        vertices: [nver, 3]
        fovy: vertical angular field of view. degree.
        aspect_ratio : width / height of field of view
        near : depth of near clipping plane
        far : depth of far clipping plane
    Returns:
        projected_vertices: [nver, 3] 
    '''
    fovy = np.deg2rad(fovy)
    top = near*np.tan(fovy)
    bottom = -top 
    right = top*aspect_ratio
    left = -right

    #-- homo
    P = np.array([[near/right, 0, 0, 0],
                 [0, near/top, 0, 0],
                 [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                 [0, 0, -1, 0]])
    vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1)))) # [nver, 4]
    projected_vertices = vertices_homo.dot(P.T)
    projected_vertices = projected_vertices/projected_vertices[:,3:]
    projected_vertices = projected_vertices[:,:3]
    projected_vertices[:,2] = -projected_vertices[:,2]

    #-- non homo. only fovy
    # projected_vertices = vertices.copy()
    # projected_vertices[:,0] = -(near/right)*vertices[:,0]/vertices[:,2]
    # projected_vertices[:,1] = -(near/top)*vertices[:,1]/vertices[:,2]
    return projected_vertices


def to_image(vertices, h, w, is_perspective = False):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis. 
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        projected_vertices: [nver, 3]  
    '''
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:,0] = image_vertices[:,0]*w/2
        image_vertices[:,1] = image_vertices[:,1]*h/2
    # move to center of image
    image_vertices[:,0] = image_vertices[:,0] + w/2
    image_vertices[:,1] = image_vertices[:,1] + h/2
    # flip vertices along y-axis.
    image_vertices[:,1] = h - image_vertices[:,1] - 1
    return image_vertices


#### -------------------------------------------2. estimate transform matrix from correspondences.
def estimate_affine_matrix_3d23d(X, Y):
    ''' Using least-squares solution 
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    '''
    X_homo = np.hstack((X, np.ones([X.shape[1],1]))) #n x 4
    P = np.linalg.lstsq(X_homo, Y)[0].T # Affine matrix. 3 x 4
    return P
    
def estimate_affine_matrix_3d22d(X, x):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. in MVGCV 
        Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
        x_homo = X_homo.dot(P_Affine)
    Args:
        X: [n, 3]. corresponding 3d points(fixed)
        x: [n, 2]. n>=4. 2d points(moving). x = PX
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    '''
    X = X.T; x = x.T
    assert(x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert(n >= 4)

    #--- 1. normalization
    # 2d points
    mean = np.mean(x, 1) # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x**2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3,3), dtype = np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean*scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1) # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3,:] - X
    average_norm = np.mean(np.sqrt(np.sum(X**2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4,4), dtype = np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean*scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n*2, 8), dtype = np.float32);
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])
 
    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype = np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))
    return P_Affine

def P2sRt(P):
    ''' decompositing camera matrix P
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation. 
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t

#Ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert(isRotationMatrix)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz

# def matrix2angle(R):
#     ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
#     Args:
#         R: (3,3). rotation matrix
#     Returns:
#         x: yaw
#         y: pitch
#         z: roll
#     '''
#     # assert(isRotationMatrix(R))

#     if R[2,0] !=1 or R[2,0] != -1:
#         x = math.asin(R[2,0])
#         y = math.atan2(R[2,1]/cos(x), R[2,2]/cos(x))
#         z = math.atan2(R[1,0]/cos(x), R[0,0]/cos(x))
        
#     else:# Gimbal lock
#         z = 0 #can be anything
#         if R[2,0] == -1:
#             x = np.pi/2
#             y = z + math.atan2(R[0,1], R[0,2])
#         else:
#             x = -np.pi/2
#             y = -z + math.atan2(-R[0,1], -R[0,2])

#     return x, y, z