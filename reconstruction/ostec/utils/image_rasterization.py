# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

from menpo3d.rasterize import (
    rasterize_barycentric_coordinate_images,
    rasterize_mesh_from_barycentric_coordinate_images)

from menpo.transform import Rotation, Translation, Scale
from menpo3d.camera import PerspectiveProjection, PerspectiveCamera
import numpy as np

def rotation_z(theta, degrees=True):
   if degrees:
       # convert to radians
       theta = theta * np.pi / 180.0
   return Rotation(np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]]),
                   skip_checks=True)

def rotation_y(theta, degrees=True):
   if degrees:
       # convert to radians
       theta = theta * np.pi / 180.0
   return Rotation(np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0],
                             [-np.sin(theta), 0, np.cos(theta)]]),
                   skip_checks=True)

def rotation_x(theta, degrees=True):
   if degrees:
       theta = np.deg2rad(theta)
   return Rotation(np.array([[ 1,             0,              0],
                             [ 0, np.cos(theta), -np.sin(theta)],
                             [ 0, np.sin(theta), np.cos(theta)]]),
                   skip_checks=True)

def perspective_camera_for_template(img_shape, focal_length_mult=2,
                                     pose_angle_deg=[0,0,0], cam_dist = 7):
    f = np.array(img_shape).max() * focal_length_mult
    rot_z = rotation_z(180 + pose_angle_deg[2])
    rot_y = rotation_y(180 + pose_angle_deg[1])
    rot_x = rotation_x(pose_angle_deg[0])
    rotation = rot_z.compose_before(rot_y).compose_before(rot_x)

    translation = Translation([0, 0, +cam_dist])
    projection = PerspectiveProjection(f, img_shape)
    return PerspectiveCamera(rotation, translation, projection)

def align_mesh_to_template(source, target, scale_corrective=1.2):
    scale = Scale((target.norm() / source.norm()) * scale_corrective,
                  n_dims=target.n_dims)
    translation = Translation(target.centre() - source.centre())
    return translation.compose_before(scale)

def rasterize_image( mesh, img_shape, pose_angle_deg=[0,0,0], cam_dist = 7, ):
    camera = perspective_camera_for_template(img_shape, pose_angle_deg=pose_angle_deg, cam_dist= cam_dist)

    # Pre-process - align the mesh roughly with the template
    # aligned_mesh = align_mesh_to_template(mesh, template).apply(mesh)

    mesh_in_img = camera.apply(mesh)

    bcs = rasterize_barycentric_coordinate_images(mesh_in_img, img_shape)
    img = rasterize_mesh_from_barycentric_coordinate_images(mesh_in_img, *bcs)
#   shape_img = rasterize_shape_image_from_barycentric_coordinate_images(mesh, *bcs)
    img.pixels = np.clip(img.pixels,0.0,1.0)
    return img, mesh_in_img.points