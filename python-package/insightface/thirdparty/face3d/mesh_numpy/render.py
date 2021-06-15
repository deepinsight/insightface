'''
functions about rendering mesh(from 3d obj to 2d image).
only use rasterization render here.
Note that:
1. Generally, render func includes camera, light, raterize. Here no camera and light(I write these in other files)
2. Generally, the input vertices are normalized to [-1,1] and cetered on [0, 0]. (in world space)
   Here, the vertices are using image coords, which centers on [w/2, h/2] with the y-axis pointing to oppisite direction.
Means: render here only conducts interpolation.(I just want to make the input flexible)

Preparation knowledge:
z-buffer: https://cs184.eecs.berkeley.edu/lecture/pipeline

Author: Yao Feng 
Mail: yaofeng1995@gmail.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time

def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

def get_point_weight(point, tri_points):
    ''' Get the weights of the position
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster.(used)
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        w0: weight of v0
        w1: weight of v1
        w2: weight of v3
     '''
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2

def rasterize_triangles(vertices, triangles, h, w):
    ''' 
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        h: height
        w: width
    Returns:
        depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle). 
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # h, w is the size of rendering
    '''
    # initial 
    depth_buffer = np.zeros([h, w]) - 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
    triangle_buffer = np.zeros([h, w], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = np.zeros([h, w, 3], dtype = np.float32)  # 
    
    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2]) # barycentric weight
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    triangle_buffer[v, u] = i
                    barycentric_weight[v, u, :] = np.array([w0, w1, w2])

    return depth_buffer, triangle_buffer, barycentric_weight


def render_colors_ras(vertices, triangles, colors, h, w, c = 3):
    ''' render mesh with colors(rasterize triangle first)
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
        c: channel
    Returns:
        image: [h, w, c]. rendering.
    '''
    assert vertices.shape[0] == colors.shape[0]

    depth_buffer, triangle_buffer, barycentric_weight = rasterize_triangles(vertices, triangles, h, w)

    triangle_buffer_flat = np.reshape(triangle_buffer, [-1]) # [h*w]
    barycentric_weight_flat = np.reshape(barycentric_weight, [-1, c]) #[h*w, c]
    weight = barycentric_weight_flat[:, :, np.newaxis] # [h*w, 3(ver in tri), 1]

    colors_flat = colors[triangles[triangle_buffer_flat, :], :] # [h*w(tri id in pixel), 3(ver in tri), c(color in ver)]
    colors_flat = weight*colors_flat # [h*w, 3, 3]
    colors_flat = np.sum(colors_flat, 1) #[h*w, 3]. add tri.

    image = np.reshape(colors_flat, [h, w, c])
    # mask = (triangle_buffer[:,:] > -1).astype(np.float32)
    # image = image*mask[:,:,np.newaxis]
    return image


def render_colors(vertices, triangles, colors, h, w, c = 3):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
    Returns:
        image: [h, w, c]. 
    '''
    assert vertices.shape[0] == colors.shape[0]
    
    # initial 
    image = np.zeros((h, w, c))
    depth_buffer = np.zeros([h, w]) - 999999.

    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]

                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    image[v, u, :] = w0*colors[tri[0], :] + w1*colors[tri[1], :] + w2*colors[tri[2], :]
    return image


def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, c = 3, mapping_type = 'nearest'):
    ''' render mesh with texture map
    Args:
        vertices: [nver], 3
        triangles: [ntri, 3]
        texture: [tex_h, tex_w, 3]
        tex_coords: [ntexcoords, 3]
        tex_triangles: [ntri, 3]
        h: height of rendering
        w: width of rendering
        c: channel
        mapping_type: 'bilinear' or 'nearest'
    '''
    assert triangles.shape[0] == tex_triangles.shape[0]
    tex_h, tex_w, _ = texture.shape

    # initial 
    image = np.zeros((h, w, c))
    depth_buffer = np.zeros([h, w]) - 999999.

    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices
        tex_tri = tex_triangles[i, :] # 3 tex indice

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                if point_depth > depth_buffer[v, u]:
                    # update depth
                    depth_buffer[v, u] = point_depth    
                    
                    # tex coord
                    tex_xy = w0*tex_coords[tex_tri[0], :] + w1*tex_coords[tex_tri[1], :] + w2*tex_coords[tex_tri[2], :]
                    tex_xy[0] = max(min(tex_xy[0], float(tex_w - 1)), 0.0); 
                    tex_xy[1] = max(min(tex_xy[1], float(tex_h - 1)), 0.0); 

                    # nearest
                    if mapping_type == 'nearest':
                        tex_xy = np.round(tex_xy).astype(np.int32)
                        tex_value = texture[tex_xy[1], tex_xy[0], :] 

                    # bilinear
                    elif mapping_type == 'bilinear':
                        # next 4 pixels
                        ul = texture[int(np.floor(tex_xy[1])), int(np.floor(tex_xy[0])), :]
                        ur = texture[int(np.floor(tex_xy[1])), int(np.ceil(tex_xy[0])), :]
                        dl = texture[int(np.ceil(tex_xy[1])), int(np.floor(tex_xy[0])), :]
                        dr = texture[int(np.ceil(tex_xy[1])), int(np.ceil(tex_xy[0])), :]

                        yd = tex_xy[1] - np.floor(tex_xy[1])
                        xd = tex_xy[0] - np.floor(tex_xy[0])
                        tex_value = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd

                    image[v, u, :] = tex_value
    return image