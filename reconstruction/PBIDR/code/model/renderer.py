import torch
import torch.nn as nn
import numpy as np
import trimesh
import os

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork


def barycentric_coordinates(p, select_vertices):

    a = select_vertices[:, 0, :]
    b = select_vertices[:, 1, :]
    c = select_vertices[:, 2, :]
    # p = point

    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(axis=1)
    d01 = (v0 * v1).sum(axis=1)
    d11 = (v1 * v1).sum(axis=1)
    d20 = (v2 * v0).sum(axis=1)
    d21 = (v2 * v1).sum(axis=1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return np.vstack([u, v, w]).T

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class AlbedoNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            dims=[512, 512, 512, 512],
            weight_norm=True,
            multires_view=4,
    ):
        super().__init__()

        dims = [3 + feature_vector_size] + dims + [3]
        embedview_fn, input_ch = get_embedder(multires_view)
        self.embedview_fn = embedview_fn
        dims[0] += (input_ch - 3)
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, feature_vectors):

        Mpoints = self.embedview_fn(points)
        x = torch.cat([Mpoints, feature_vectors], dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x

class SpecularNetwork(nn.Module):
    def __init__(
            self,
            dims=[256, 256, 256],
            weight_norm=True,
            multires_view=4
    ):
        super().__init__()
        dims = [3 + 3] + dims + [1]

        embedview_fn, input_ch = get_embedder(multires_view)
        self.embedview_fn = embedview_fn
        dims[0] += (input_ch - 3)
        dims[0] += (input_ch - 3)
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, normals, view_dirs):

        Mview_dirs = self.embedview_fn(view_dirs)
        Mnormals= self.embedview_fn(normals)
        x = torch.cat([Mview_dirs, Mnormals], dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x
    def optimaize(self):
        return

class DiffuseNetwork(nn.Module):
    def __init__(
            self,
            dims=[256, 256, 256],
            weight_norm=True,
            multires_view=6,
    ):
        super().__init__()

        dims = [3] + dims + [1]
        embedview_fn, input_ch = get_embedder(multires_view)
        self.embedview_fn = embedview_fn
        dims[0] += (input_ch - 3)
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, normals):

        Mnormals = self.embedview_fn(normals)
        x = Mnormals

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x

class IFNetwork(nn.Module):
    def __init__(self, conf, id, datadir):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        # self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))

        self.diffuse_network = DiffuseNetwork(**conf.get_config('diffuse_network'))
        self.specular_network = SpecularNetwork(**conf.get_config('specular_network'))
        self.albedo_network = AlbedoNetwork(self.feature_vector_size, **conf.get_config('albedo_network'))

        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.mesh = trimesh.load_mesh('{0}/mesh.obj'.format(os.path.join('../data', datadir, 'scan{0}'.format(id))),
                                      process=False, use_embree=True)
        self.faces = self.mesh.faces
        self.vertex_normals = np.array(self.mesh.vertex_normals)
        self.vertices = np.array(self.mesh.vertices)
        print('Loaded Mesh')

    def forward(self, input):

        # Parse model input
        points_predicted = None

        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)
        points_normal = self.implicit_network.gradient(points)
        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        ray_dirs_np = ray_dirs.cpu().numpy()
        cam_loc_np = np.concatenate([cam_loc.cpu().numpy()] * len(ray_dirs_np), axis=0)
        # points_mesh_ray: may have the more points than surface mask points,
        # Need an Index for the Points_Mesh_Ray
        points_mesh_ray, index_ray, index_tri = self.mesh.ray.intersects_location(ray_origins=cam_loc_np,
                                                                                  ray_directions=ray_dirs_np,
                                                                                  multiple_hits=False)
        # Index ray: total 2048 / ~1200
        MeshRay_mask = torch.tensor([True if i in index_ray else False for i in range(len(cam_loc_np))], dtype=torch.bool).to(points.device)
        network_object_mask = network_object_mask & MeshRay_mask

        if self.training:

            surface_mask = network_object_mask & object_mask

            listA = surface_mask.cpu().detach().numpy()
            A = [int(a) for a in listA]
            AA = [i for i, a in enumerate(A) if a == 1] # surface mask 的 index
            MeshRay_Index = np.array([i for i, a in enumerate(index_ray) if a in AA], dtype=int)

            face_points_index = self.faces[index_tri][MeshRay_Index]
            select_vertex_normals = self.vertex_normals[face_points_index]
            select_vertices = self.vertices[face_points_index]

            points_mesh_ray = points_mesh_ray[MeshRay_Index]
            bcoords = barycentric_coordinates(points_mesh_ray, select_vertices)
            resampled_normals = np.sum(np.expand_dims(bcoords, -1) * select_vertex_normals, 1)

            # Mesh Pull
            resampled_normals = torch.tensor(resampled_normals).to(points)
            points_mesh_ray = torch.tensor(points_mesh_ray).to(points)
            sdf_points_mesh_ray = self.implicit_network(points_mesh_ray)[:, 0:1]
            g_points_mesh_ray = self.implicit_network.gradient(points_mesh_ray)
            points_predicted = points_mesh_ray - g_points_mesh_ray.squeeze() * sdf_points_mesh_ray

            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

            listA = surface_mask.cpu().detach().numpy()
            A = [int(a) for a in listA]
            AA = [i for i, a in enumerate(A) if a == 1]  # surface mask 的 index
            MeshRay_Index = np.array([i for i, a in enumerate(index_ray) if a in AA], dtype=int)

            face_points_index = self.faces[index_tri][MeshRay_Index]
            select_vertex_normals = self.vertex_normals[face_points_index]
            select_vertices = self.vertices[face_points_index]

            points_mesh_ray = points_mesh_ray[MeshRay_Index]
            bcoords = barycentric_coordinates(points_mesh_ray, select_vertices)
            resampled_normals = np.sum(np.expand_dims(bcoords, -1) * select_vertex_normals, 1)
            resampled_normals = torch.tensor(resampled_normals).to(points)

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        diffuse_values = torch.ones_like(points).float().cuda()
        specular_values = torch.ones_like(points).float().cuda()
        albedo_values = torch.ones_like(points).float().cuda()

        if differentiable_surface_points.shape[0] > 0:

            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view, resampled_normals)
            diffuse_values[surface_mask] = self.get_diffuse_value(differentiable_surface_points, view, resampled_normals)

            specular_values[surface_mask] = self.get_specular_value(differentiable_surface_points, view)
            albedo_values[surface_mask] = self.get_albedo_value(differentiable_surface_points, view)

        output = {
            'points': points,
            'points_pre': points_predicted,
            'points_mesh_ray_gt': points[surface_mask],
            'points_mesh_ray_normals': resampled_normals,
            'surface_normals': points_normal[surface_mask].reshape([-1, 3]),

            'rgb_values': rgb_values,
            'diffuse_values': diffuse_values,
            'specular_values': specular_values,
            'albedo_values': albedo_values,

            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta
        }

        return output

    def get_rbg_value(self, points, view_dirs, diffuse_normals):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]
        feature_vectors = output[:, 1:]

        diffuse_shading = self.diffuse_network(diffuse_normals)
        specular_shading = self.specular_network(normals, view_dirs)
        albedo = self.albedo_network(points, feature_vectors)

        diffuse_shading = (diffuse_shading + 1.) / 2.
        specular_shading = (specular_shading + 1.) / 2.
        albedo = (albedo + 1.) / 2.

        rgb_vals = diffuse_shading * albedo + specular_shading
        rgb_vals = (rgb_vals * 2.) - 1.

        return rgb_vals

    def get_diffuse_value(self, points, view_dirs, diffuse_normals):

        diffuse_shading = self.diffuse_network(diffuse_normals)
        return diffuse_shading.expand([-1, 3])

    def get_albedo_value(self, points, view_dirs):
        output = self.implicit_network(points)
        feature_vectors = output[:, 1:]
        albedo = self.albedo_network(points, feature_vectors)

        return albedo

    def get_specular_value(self, points, view_dirs):
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        specular_shading = self.specular_network(normals, view_dirs)
        return specular_shading.expand([-1, 3])