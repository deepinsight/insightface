# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

""" Renders 3D faces in python with lambertian shading
    Author: Stylianos Ploumpis """

import numpy as np
from menpo.transform import UniformScale, Translation


def l2_normalize(x, axis=0, epsilon=1e-12):
    """
    Transforms an `ndarray` to have a unit l2 norm along
    a given direction.
    ----------
    x : `ndarray`
        The array to be transformed.
    axis : `int`
        The axis that will be l2 unit normed.
    epsilon: `float`
        A small value such as to avoid division by zero.

    Returns
    -------
    x : (D,) `ndarray`
        The transformed array.
    """
    return x / np.maximum(np.linalg.norm(x, axis=axis), epsilon)


def mesh_in_unit_sphere(mesh):
    scale = UniformScale(1 / mesh.norm(), mesh.n_dims)
    translation = Translation(-scale.apply(mesh).centre())
    return translation.compose_after(scale)


def lambertian_shading(mesh, diffuse_colour=0.4,
                       albedo_weighting=0.6, ambient_colour=0.2,
                       light_positions=((45, 45, 1), (30, 60, 10),(30, 60, 0.5),(0, 0, 1))):

    diffuse_colour = np.asarray(diffuse_colour)
    light_positions = l2_normalize(np.asarray(light_positions).reshape(-1, 3),
                                   axis=0)

    unit_transform = mesh_in_unit_sphere(mesh)
    mesh = unit_transform.apply(mesh)

    light_directions = l2_normalize(light_positions.reshape(-1, 1, 3) -
                                    mesh.points[None, ...], axis=0)

    # Calculate the lambertian reflectance for each light source.
    # This will be an `ndarray` of shape(num_light_sources, num_vertices)
    lambertian = np.sum(light_directions *
                        mesh.vertex_normals()[None, ...], 2)[..., None]

    # Sum up the contribution of all the light sources and multiply by the
    # diffusion colour.
    lambertian = lambertian.sum(0) * diffuse_colour + ambient_colour

    mesh.colours[...] = np.clip(mesh.colours * albedo_weighting +
                                lambertian * (1 - albedo_weighting),
                                0, 1)

    return unit_transform.pseudoinverse().apply(mesh)