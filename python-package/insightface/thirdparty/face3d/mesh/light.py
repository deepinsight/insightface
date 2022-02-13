'''
Functions about lighting mesh(changing colors/texture of mesh).
1. add light to colors/texture (shade each vertex)
2. fit light according to colors/texture & image.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .cython import mesh_core_cython

def get_normal(vertices, triangles):
    ''' calculate normal direction in each vertex
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3]
    '''
    pt0 = vertices[triangles[:, 0], :] # [ntri, 3]
    pt1 = vertices[triangles[:, 1], :] # [ntri, 3]
    pt2 = vertices[triangles[:, 2], :] # [ntri, 3]
    tri_normal = np.cross(pt0 - pt1, pt0 - pt2) # [ntri, 3]. normal of each triangle

    normal = np.zeros_like(vertices, dtype = np.float32).copy() # [nver, 3]
    # for i in range(triangles.shape[0]):
    #     normal[triangles[i, 0], :] = normal[triangles[i, 0], :] + tri_normal[i, :]
    #     normal[triangles[i, 1], :] = normal[triangles[i, 1], :] + tri_normal[i, :]
    #     normal[triangles[i, 2], :] = normal[triangles[i, 2], :] + tri_normal[i, :]
    mesh_core_cython.get_normal_core(normal, tri_normal.astype(np.float32).copy(), triangles.copy(), triangles.shape[0])

    # normalize to unit length
    mag = np.sum(normal**2, 1) # [nver]
    zero_ind = (mag == 0)
    mag[zero_ind] = 1;
    normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))

    normal = normal/np.sqrt(mag[:,np.newaxis])

    return normal

# TODO: test
def add_light_sh(vertices, triangles, colors, sh_coeff):
    ''' 
    In 3d face, usually assume:
    1. The surface of face is Lambertian(reflect only the low frequencies of lighting)
    2. Lighting can be an arbitrary combination of point sources
    --> can be expressed in terms of spherical harmonics(omit the lighting coefficients)
    I = albedo * (sh(n) x sh_coeff)
    
    albedo: n x 1
    sh_coeff: 9 x 1
    Y(n) = (1, n_x, n_y, n_z, n_xn_y, n_xn_z, n_yn_z, n_x^2 - n_y^2, 3n_z^2 - 1)': n x 9 
    # Y(n) = (1, n_x, n_y, n_z)': n x 4

    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        colors: [nver, 3] albedo
        sh_coeff: [9, 1] spherical harmonics coefficients

    Returns:
        lit_colors: [nver, 3]
    '''
    assert vertices.shape[0] == colors.shape[0]
    nver = vertices.shape[0]
    normal = get_normal(vertices, triangles) # [nver, 3]
    sh = np.array((np.ones(nver), n[:,0], n[:,1], n[:,2], n[:,0]*n[:,1], n[:,0]*n[:,2], n[:,1]*n[:,2], n[:,0]**2 - n[:,1]**2, 3*(n[:,2]**2) - 1)) # [nver, 9]
    ref = sh.dot(sh_coeff) #[nver, 1]
    lit_colors = colors*ref
    return lit_colors


def add_light(vertices, triangles, colors, light_positions = 0, light_intensities = 0):
    ''' Gouraud shading. add point lights.
    In 3d face, usually assume:
    1. The surface of face is Lambertian(reflect only the low frequencies of lighting)
    2. Lighting can be an arbitrary combination of point sources
    3. No specular (unless skin is oil, 23333)

    Ref: https://cs184.eecs.berkeley.edu/lecture/pipeline    
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        light_positions: [nlight, 3] 
        light_intensities: [nlight, 3]
    Returns:
        lit_colors: [nver, 3]
    '''
    nver = vertices.shape[0]
    normals = get_normal(vertices, triangles) # [nver, 3]

    # ambient
    # La = ka*Ia

    # diffuse
    # Ld = kd*(I/r^2)max(0, nxl)
    direction_to_lights = vertices[np.newaxis, :, :] - light_positions[:, np.newaxis, :] # [nlight, nver, 3]
    direction_to_lights_n = np.sqrt(np.sum(direction_to_lights**2, axis = 2)) # [nlight, nver]
    direction_to_lights = direction_to_lights/direction_to_lights_n[:, :, np.newaxis]
    normals_dot_lights = normals[np.newaxis, :, :]*direction_to_lights # [nlight, nver, 3]
    normals_dot_lights = np.sum(normals_dot_lights, axis = 2) # [nlight, nver]
    diffuse_output = colors[np.newaxis, :, :]*normals_dot_lights[:, :, np.newaxis]*light_intensities[:, np.newaxis, :]
    diffuse_output = np.sum(diffuse_output, axis = 0) # [nver, 3]
    
    # specular
    # h = (v + l)/(|v + l|) bisector
    # Ls = ks*(I/r^2)max(0, nxh)^p
    # increasing p narrows the reflectionlob

    lit_colors = diffuse_output # only diffuse part here.
    lit_colors = np.minimum(np.maximum(lit_colors, 0), 1)
    return lit_colors



## TODO. estimate light(sh coeff)
## -------------------------------- estimate. can not use now. 
def fit_light(image, vertices, colors, triangles, vis_ind, lamb = 10, max_iter = 3):
    [h, w, c] = image.shape

    # surface normal
    norm = get_normal(vertices, triangles)
    
    nver = vertices.shape[1]

    # vertices --> corresponding image pixel
    pt2d = vertices[:2, :]

    pt2d[0,:] = np.minimum(np.maximum(pt2d[0,:], 0), w - 1)
    pt2d[1,:] = np.minimum(np.maximum(pt2d[1,:], 0), h - 1)
    pt2d = np.round(pt2d).astype(np.int32) # 2 x nver

    image_pixel = image[pt2d[1,:], pt2d[0,:], :] # nver x 3
    image_pixel = image_pixel.T # 3 x nver

    # vertices --> corresponding mean texture pixel with illumination
    # Spherical Harmonic Basis
    harmonic_dim = 9
    nx = norm[0,:];
    ny = norm[1,:];
    nz = norm[2,:];
    harmonic = np.zeros((nver, harmonic_dim))

    pi = np.pi
    harmonic[:,0] = np.sqrt(1/(4*pi)) * np.ones((nver,));
    harmonic[:,1] = np.sqrt(3/(4*pi)) * nx;
    harmonic[:,2] = np.sqrt(3/(4*pi)) * ny;
    harmonic[:,3] = np.sqrt(3/(4*pi)) * nz;
    harmonic[:,4] = 1/2. * np.sqrt(3/(4*pi)) * (2*nz**2 - nx**2 - ny**2);
    harmonic[:,5] = 3 * np.sqrt(5/(12*pi)) * (ny*nz);
    harmonic[:,6] = 3 * np.sqrt(5/(12*pi)) * (nx*nz);
    harmonic[:,7] = 3 * np.sqrt(5/(12*pi)) * (nx*ny);
    harmonic[:,8] = 3/2. * np.sqrt(5/(12*pi)) * (nx*nx - ny*ny);
    
    '''
    I' = sum(albedo * lj * hj) j = 0:9 (albedo = tex)
    set A = albedo*h (n x 9)
        alpha = lj (9 x 1)
        Y = I (n x 1)
        Y' = A.dot(alpha)

    opt function:
        ||Y - A*alpha|| + lambda*(alpha'*alpha)
    result:
        A'*(Y - A*alpha) + lambda*alpha = 0
        ==>
        (A'*A*alpha - lambda)*alpha = A'*Y
        left: 9 x 9
        right: 9 x 1
    '''
    n_vis_ind = len(vis_ind)
    n = n_vis_ind*c

    Y = np.zeros((n, 1))
    A = np.zeros((n, 9))
    light = np.zeros((3, 1))

    for k in range(c):
        Y[k*n_vis_ind:(k+1)*n_vis_ind, :] = image_pixel[k, vis_ind][:, np.newaxis]
        A[k*n_vis_ind:(k+1)*n_vis_ind, :] = texture[k, vis_ind][:, np.newaxis] * harmonic[vis_ind, :]
        Ac = texture[k, vis_ind][:, np.newaxis]
        Yc = image_pixel[k, vis_ind][:, np.newaxis]
        light[k] = (Ac.T.dot(Yc))/(Ac.T.dot(Ac))

    for i in range(max_iter):

        Yc = Y.copy()
        for k in range(c):
            Yc[k*n_vis_ind:(k+1)*n_vis_ind, :]  /= light[k]

        # update alpha
        equation_left = np.dot(A.T, A) + lamb*np.eye(harmonic_dim); # why + ?
        equation_right = np.dot(A.T, Yc) 
        alpha = np.dot(np.linalg.inv(equation_left), equation_right)

        # update light
        for k in range(c):
            Ac = A[k*n_vis_ind:(k+1)*n_vis_ind, :].dot(alpha)
            Yc = Y[k*n_vis_ind:(k+1)*n_vis_ind, :]
            light[k] = (Ac.T.dot(Yc))/(Ac.T.dot(Ac))

    appearance = np.zeros_like(texture)
    for k in range(c):
        tmp = np.dot(harmonic*texture[k, :][:, np.newaxis], alpha*light[k])
        appearance[k,:] = tmp.T

    appearance = np.minimum(np.maximum(appearance, 0), 1)

    return appearance

