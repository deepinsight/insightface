# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import numpy as np
import math

"""Collection of functions to adapt GANFit camera parameters"""

GANFIT_CAMERA_CONSTANTS = [np.array([[0.0, 0.0, 6.0]], dtype=np.float32),
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                    np.array([20.], dtype=np.float32)]

def look_at(eye, center, world_up):
    """Computes camera viewing matrices.
    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).
    Args:
    eye: 2-D float32 tensor with shape [batch_size, 3] containing the XYZ world
        space position of the camera.
    center: 2-D float32 tensor with shape [batch_size, 3] containing a position
        along the center of the camera's gaze.
    world_up: 2-D float32 tensor with shape [batch_size, 3] specifying the
        world's up direction; the output camera will have no tilt with respect
        to this direction.
    Returns:
    A [batch_size, 4, 4] float tensor containing a right-handed camera
    extrinsics matrix that maps points from world space to points in eye space.
    """
    batch_size = center.shape[0]
    vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = np.linalg.norm(forward, axis=1, keepdims=True)
    forward = np.divide(forward, forward_norm)

    to_side = np.cross(forward, world_up)
    to_side_norm = np.linalg.norm(to_side, axis=1, keepdims=True)
    to_side = np.divide(to_side, to_side_norm)
    cam_up = np.cross(to_side, forward)

    w_column = np.array(
        batch_size * [[0., 0., 0., 1.]], dtype=np.float32)  # [batch_size, 4]
    w_column = np.reshape(w_column, [batch_size, 4, 1])
    view_rotation = np.stack(
        [to_side, cam_up, -forward,
         np.zeros_like(to_side, dtype=np.float32)],
        axis=1)  # [batch_size, 4, 3] matrix
    view_rotation = np.concatenate(
        [view_rotation, w_column], axis=2)  # [batch_size, 4, 4]

    identity_batch = np.tile(np.expand_dims(np.eye(3), 0), [batch_size, 1, 1])
    view_translation = np.concatenate([identity_batch, np.expand_dims(-eye, 2)], 2)
    view_translation = np.concatenate(
        [view_translation,
         np.reshape(w_column, [batch_size, 1, 4])], 1)
    camera_matrices = np.matmul(view_rotation, view_translation)
    return camera_matrices

def perspective(aspect_ratio, fov_y, near_clip, far_clip):
    """Computes perspective transformation matrices.
    Functionality mimes gluPerspective (third_party/GL/glu/include/GLU/glu.h).
    Args:
      aspect_ratio: float value specifying the image aspect ratio (width/height).
      fov_y: 1-D float32 Tensor with shape [batch_size] specifying output vertical
          field of views in degrees.
      near_clip: 1-D float32 Tensor with shape [batch_size] specifying near
          clipping plane distance.
      far_clip: 1-D float32 Tensor with shape [batch_size] specifying far clipping
          plane distance.
    Returns:
      A [batch_size, 4, 4] float tensor that maps from right-handed points in eye
      space to left-handed points in clip space.
    """
    focal_lengths_y = 1.0 / np.tan(fov_y * (math.pi / 360.0))
    depth_range = far_clip - near_clip
    p_22 = -(far_clip + near_clip) / depth_range
    p_23 = -2.0 * (far_clip * near_clip / depth_range)

    zeros = np.zeros_like(p_23, dtype=np.float32)
    # pyformat: disable
    perspective_transform = np.concatenate(
      [
          focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
          zeros, focal_lengths_y, zeros, zeros,
          zeros, zeros, p_22, p_23,
          zeros, zeros, -np.ones_like(p_23, dtype=np.float32), zeros
      ], axis=0)
    # pyformat: enable
    perspective_transform = np.reshape(perspective_transform, [4, 4, -1])
    return np.transpose(perspective_transform, [2, 0, 1])

def euler_matrices(angles):
  """Computes a XYZ Tait-Bryan (improper Euler angle) rotation.
  Returns 4x4 matrices for convenient multiplication with other transformations.
  Args:
    angles: a [batch_size, 3] tensor containing X, Y, and Z angles in radians.
  Returns:
    a [batch_size, 4, 4] tensor of matrices.
  """
  s = np.sin(angles)
  c = np.cos(angles)
  # Rename variables for readability in the matrix definition below.
  c0, c1, c2 = (c[:, 0], c[:, 1], c[:, 2])
  s0, s1, s2 = (s[:, 0], s[:, 1], s[:, 2])

  zeros = np.zeros_like(s[:, 0])
  ones = np.ones_like(s[:, 0])

  # pyformat: disable
  flattened = np.concatenate(
      [
          c2 * c1, c2 * s1 * s0 - c0 * s2, s2 * s0 + c2 * c0 * s1, zeros,
          c1 * s2, c2 * c0 + s2 * s1 * s0, c0 * s2 * s1 - c2 * s0, zeros,
          -s1, c1 * s0, c1 * c0, zeros,
          zeros, zeros, zeros, ones
      ],
      axis=0)
  # pyformat: enable
  reshaped = np.reshape(flattened, [4, 4, -1])
  return np.transpose(reshaped, [2, 0, 1])

def transform_homogeneous(matrices, vertices):
  """Applies batched 4x4 homogenous matrix transformations to 3-D vertices.
  The vertices are input and output as as row-major, but are interpreted as
  column vectors multiplied on the right-hand side of the matrices. More
  explicitly, this function computes (MV^T)^T.
  Vertices are assumed to be xyz, and are extended to xyzw with w=1.
  Args:
    matrices: a [batch_size, 4, 4] tensor of matrices.
    vertices: a [batch_size, N, 3] tensor of xyz vertices.
  Returns:
    a [batch_size, N, 4] tensor of xyzw vertices.
  Raises:
    ValueError: if matrices or vertices have the wrong number of dimensions.
  """
  homogeneous_coord = np.ones(
      [np.shape(vertices)[0], np.shape(vertices)[1], 1], dtype=np.float32)
  vertices_homogeneous = np.concatenate([vertices, homogeneous_coord], 2)

  return np.matmul(vertices_homogeneous[0], matrices[0].T)


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
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


# Ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
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
    assert (isRotationMatrix)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    # rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    # return rx, ry, rz
    return x, y, z

def get_camera_matrices(camera_params):
    # defaults from ganfit

    camera_position = camera_params[0] + GANFIT_CAMERA_CONSTANTS[0]
    camera_lookat = camera_params[1] + GANFIT_CAMERA_CONSTANTS[1]
    camera_up =  camera_params[2] + GANFIT_CAMERA_CONSTANTS[2]

    return look_at(camera_position, camera_lookat, camera_up)

def get_pose(camera_params):
    s, R, t = P2sRt(get_camera_matrices(camera_params)[0])
    return matrix2angle(R)  # pitch:%.2f,\n yaw:%.2f \n , roll:%.2f \n

def apply_camera_only3d(vertices, camera_params):

    camera_matrices = get_camera_matrices(camera_params)

    clip_space_vertices = transform_homogeneous(camera_matrices, np.array([vertices]))[:, 0:3]

    return clip_space_vertices


def apply_camera(camera_params, tmesh, fitting, ganfit_image_size = 512):
    vertices = tmesh.points

    # defaults from ganfit
    image_width = ganfit_image_size
    image_height = ganfit_image_size
    near_clip = np.array([0.01])
    far_clip = np.array([10.0])
    fov_y =  camera_params[3] + GANFIT_CAMERA_CONSTANTS[3]

    camera_matrices = get_camera_matrices(camera_params)
    perspective_transforms = perspective(image_width / image_height, fov_y, near_clip, far_clip)
    clip_space_transforms = np.matmul(perspective_transforms, camera_matrices)

    clip_space_vertices = transform_homogeneous(clip_space_transforms, np.array([vertices]))

    _MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD = 1e-6
    clip_space_points_w = np.maximum(
        np.abs(clip_space_vertices[:, 3:4]),
        _MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD) * np.sign(
        clip_space_vertices[:, 3:4])
    normalized_device_coordinates = (
            clip_space_vertices[:, 0:3] / clip_space_points_w)

    normalized_device_coordinates = ((normalized_device_coordinates[:, 0:2] * [1, -1]) / 2 + 0.5) * [image_width,
                                                                                                     image_height]

    dense_lms_org = fitting['transformation'].apply(normalized_device_coordinates[:, ::-1])[:, ::-1]

    dense_lms_org /= [1024, 1024]
    dense_lms_org[:, 1] = 1 - dense_lms_org[:, 1]

    return dense_lms_org

