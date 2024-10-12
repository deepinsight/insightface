# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import PIL.Image
from skimage.morphology import binary_dilation,  disk
from skimage.filters import gaussian
from scipy.interpolate import NearestNDInterpolator
import cv2 as cv
from utils.shading import lambertian_shading
from menpo3d.rasterize import rasterize_mesh
from menpo.shape import TriMesh, TexturedTriMesh, ColouredTriMesh
from menpo.image import Image
from utils.image_rasterization import *
import menpo.io as mio
import os

gold = (244/256,178/256,88/256)
green = (26/256,140/256,57/256)
blue = (24/256,148/256,187/256)
red = (255/256,51/256,51/256)
purple = (102/256,0/256,204/256)
pink = (255/256,21/256,255/256)

def perspective_camera(img_shape, focal_length_mult=2,
                                     pose_angle_deg=[0,0,0], translation=4):
    f = np.array(img_shape).max() * focal_length_mult
    rot_z = rotation_z(180 + pose_angle_deg[2])
    rot_y = rotation_y(180 + pose_angle_deg[1])
    rot_x = rotation_x(pose_angle_deg[0])
    rotation = rot_z.compose_before(rot_y).compose_before(rot_x)

    translation = Translation([0, 0, translation])
    projection = PerspectiveProjection(f, img_shape)
    return PerspectiveCamera(rotation, translation, projection)

def rasterize_mesh_at_template(mesh, img_shape=(1024,1024), pose_angle_deg=[0, 0, 0], translation=4, shaded=True, **kwargs):
    camera = perspective_camera(img_shape, pose_angle_deg=pose_angle_deg, translation=translation)
    if shaded:
        mesh = lambertian_shading(mesh, **kwargs)
    mesh.points[mesh.points == 0] = None
    return rasterize_mesh(camera.apply(mesh), img_shape)

def export_tmesh(tmesh, path, pose_angle_deg=[0, 30, 0], translation=4):
    render = rasterize_mesh_at_template(tmesh, pose_angle_deg=pose_angle_deg, shaded=False, translation=translation)
    mio.export_image(render, path, overwrite=True)

def export_shape(tmesh, path, pose_angle_deg=[0, 30, 0]):
    mesh = ColouredTriMesh(tmesh.points, tmesh.trilist,
                           np.tile(np.array(blue), [len(tmesh.points), 1]))
    render = rasterize_mesh_at_template(mesh, pose_angle_deg=pose_angle_deg, shaded=True)
    mio.export_image(render, path, overwrite=True)

def uv_color_normalize(img_uv_src, angle_uv_src, img_uv, angle_uv):
    intersection = (angle_uv_src.pixels[0] < 0.6) & (angle_uv_src.pixels[0] > 0.3) & (angle_uv.pixels[0] > 0.7)
    if np.any(intersection):
        target_dist = img_uv_src.pixels[:, intersection]
        target_mean = np.mean(target_dist, 1).reshape([3, -1])
        targed_std = np.std(target_dist, 1).reshape([3, -1])

        source_dist = img_uv.pixels[:, intersection]
        source_mean = np.mean(source_dist, 1).reshape([3, -1])
        source_std = np.std(source_dist, 1).reshape([3, -1])

        temp = ((img_uv.pixels.reshape([3, -1]) - source_mean) / source_std) * targed_std + target_mean
        return Image(temp.reshape(img_uv.pixels.shape))
    else:
        return img_uv

def uv_stiching(img_uv_list, angle_uv_list, smoothing_sigma = 10, seamless_clone=False):
    max_ind = np.argmax(np.array(angle_uv_list).mean(axis=1), axis=0)
    mask_out_all = np.max(np.array(angle_uv_list).mean(axis=1),axis=0) ==-1
    max_ind[mask_out_all] = len(img_uv_list) -1
    all_uvs = np.clip(np.array(img_uv_list), 0, 1)

    max_ind_one_hot = np.zeros((max_ind.size, len(img_uv_list)))
    max_ind_one_hot[np.arange(max_ind.size), max_ind.flatten()] = 1
    max_ind_one_hot = max_ind_one_hot.reshape(max_ind.shape + (-1,))

    if smoothing_sigma>0:
        max_ind_one_hot_g = gaussian(max_ind_one_hot, sigma=smoothing_sigma, multichannel=True, mode='reflect')
    else:
        max_ind_one_hot_g = max_ind_one_hot

    max_ind_one_hot = np.tile(max_ind_one_hot, [3, 1, 1, 1])
    max_ind_one_hot = np.transpose(max_ind_one_hot, [3, 0, 1, 2])

    max_ind_one_hot_g = np.tile(max_ind_one_hot_g, [3, 1, 1, 1])
    max_ind_one_hot_g = np.transpose(max_ind_one_hot_g, [3, 0, 1, 2])

    if seamless_clone:
        all_uvs_uint8 = (np.transpose(all_uvs,[0,2,3,1])*255).astype(np.uint8)
        max_ind_one_uint8 = (np.transpose(1 - max_ind_one_hot,[0,2,3,1])*255).astype(np.uint8)
        max_ind_one_uint8_g = (np.transpose(1 - max_ind_one_hot_g,[0,2,3,1])*255).astype(np.uint8)

        dst = (Image(np.sum(all_uvs * max_ind_one_hot_g, axis=0)).pixels_with_channels_at_back()*255).astype(np.uint8)
        for i, (src, mask, mask_g) in enumerate(zip(all_uvs_uint8[1:], max_ind_one_uint8[1:], max_ind_one_uint8_g[1:])):
            mask_inv = 255-mask.copy()
            mask_inv[:,0:5] = 255
            mask_inv[:, -6:-1] = 255
            mask_inv[0:5,:] = 255
            mask_inv[-6:-1,:] = 255
            mask_inv_dilated = binary_dilation(mask_inv.astype(np.bool)[:,:,0], disk(50)).astype(np.uint8)*255
            mask_inv_dilated = np.transpose(np.tile(mask_inv_dilated,[3,1,1]),[1,2,0])

            mask_inv_dilated_g = gaussian(mask_inv_dilated/255, sigma=100, multichannel=True, mode='reflect')
            mixed = (src * mask_inv_dilated_g + dst * (1-mask_inv_dilated_g)).astype(np.uint8)

            im_clone = cv.seamlessClone(dst, mixed, mask_inv, (512,512), cv.NORMAL_CLONE)
            dst = (im_clone * (1-mask_g/255) +  dst* (mask_g/255)).astype(np.uint8)
        final_uv = Image(np.transpose(dst.astype(np.float32)/255,[2,0,1]))
    else:
        final_uv = Image(np.sum(all_uvs * max_ind_one_hot_g, axis=0))

    final_uv.pixels = np.clip(final_uv.pixels,0.0, 1.0)
    return final_uv, max_ind_one_hot

def im_menpo2PIL(menpo_im):
    return PIL.Image.fromarray((menpo_im.pixels_with_channels_at_back() * 255).astype(np.uint8))

def im_PIL2menpo(pil_im):
    return Image.init_from_channels_at_back(np.array(pil_im).astype(np.float32) / 255)

def fill_UV(UV):
    mask = np.sum(UV.pixels, 0) == 0
    xx, yy = np.meshgrid(np.arange(UV.shape[1]), np.arange(UV.shape[0]))
    xym = np.vstack((np.ravel(xx[~mask]), np.ravel(yy[~mask]))).T
    data = UV.pixels[:, ~mask]
    for i in range(3):
        interp = NearestNDInterpolator(xym, data[i])
        result = interp(np.ravel(xx[mask]), np.ravel(yy[mask]))
        UV.pixels[i, mask] = result
    return UV

from keras.utils import get_file
import bz2
def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def fix_obj(fp):
    os.path.dirname(fp)
    template = """# Produced by Dimensional Imaging OBJ exporter
# http://www.di3d.com
#
#
newmtl merged_material
Ka  0.5 0.5 0.5
Kd  0.5 0.5 0.5
Ks  0.47 0.47 0.47
d 1
Ns 0
illum 2
map_Kd {}.png
#
#
# EOF""".format(os.path.splitext(os.path.basename(fp))[0])
    with open(os.path.join(os.path.dirname(fp), os.path.splitext(os.path.basename(fp))[0] + '.mtl'), 'w') as f:
        f.write(template)

    with open(fp, 'r+')  as f:
        content = f.read()
        f.seek(0, 0)
        f.write('mtllib ' + os.path.splitext(os.path.basename(fp))[0] + '.mtl' + '\n' + content)

