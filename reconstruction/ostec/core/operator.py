# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

from utils.align2stylegan import align_im2stylegan, align_mesh2stylegan
from core.projection_handler import Projection_Handler
from skimage.morphology import remove_small_holes
import time
from utils.ganfit_camera import apply_camera_only3d, get_pose
from utils.utils import *
from core.arcface_handler import Arcface_Handler

from utils import generate_heatmap

class Face:
    def __init__(self, tmesh,
            tmesh_masked,
            tmesh_rotated,
            img_uv_src,
            angle_uv_src,
            angle_uv_list,
            img_uv_list,
            view_angle_src,
            id_features,
            exclude_mask,
            is_profile,
            mode):
        self.tmesh = tmesh
        self.tmesh_masked = tmesh_masked
        self.tmesh_rotated = tmesh_rotated
        self.img_uv_src = img_uv_src
        self.angle_uv_src = angle_uv_src
        self.angle_uv_list = angle_uv_list
        self.img_uv_list = img_uv_list
        self.view_angle_src = view_angle_src
        self.id_features = id_features
        self.is_profile = is_profile
        self.exclude_mask = exclude_mask

        if mode == 'auto':
            if is_profile:
                self.mode = 'hard'
            else:
                self.mode = 'soft'
        else:
            self.mode = mode

    def rotation_dict(self):
        if self.mode == 'soft':
            return {
                'bottom': [-30, 0, 0],
                'bottom_left': [-15, -30, 0],
                'bottom_right': [-15, 30, 0],
                'left': [5, -40, 0],
                'right': [5, 40, 0],
            }
        elif self.mode == 'hard':
            return {
                'front': [0, 0, 0],
                'bottom': [-30, 0, 0],
                'bottom_left': [-15, -30, 0],
                'bottom_right': [-15, 30, 0],
                'left': [5, -40, 0],
                'right': [5, 40, 0],
            }
        else:
            raise Exception('Unknown mode!')

    def coef_dict(self):
        if self.mode == 'soft':
            return {
                'bottom': 1.3,
                'bottom_left': 1,
                'bottom_right': 1,
                'left': 1,
                'right': 1,
                'src': 2,
            }
        elif self.mode == 'hard':
            return {
                'front': 2,
                'bottom': 1.3,
                'bottom_left': 1,
                'bottom_right': 1,
                'left': 1,
                'right': 1,
                'src': 0.1,
            }
        else:
            raise Exception('Unknown mode!')


class Operator:
    def __init__(self, args):
        self.tcoords_full = mio.import_pickle('models/topology/tcoords_full.pkl')
        self.tcoords = mio.import_pickle('models/topology/tcoords_alex.pkl')
        self.mask = mio.import_pickle('models/topology/mask_full2crop.pkl') | True
        self.tight_mask = mio.import_pickle('models/topology/mask_full2tightcrop.pkl')
        self.template = mio.import_pickle('models/topology/all_all_all_crop_mean.pkl')
        self.lms_ind = mio.import_pickle('models/topology/all_all_all_lands_ids.pkl')
        self.img_shape = [1024, 1024] # 2048

        self.uv_shape = [1024, 1536]
        uv_mesh = self.tcoords.copy().points[:, ::-1]
        uv_mesh[:, 0] = 1 - uv_mesh[:, 0]
        uv_mesh *= self.uv_shape
        self.uv_mesh = np.concatenate([uv_mesh, uv_mesh[:, 0:1] * 0], 1)
        self.uv_trilist = mio.import_pickle('models/topology/trilist_full.pkl') #self.template.trilist

        self.args = args
        self.mode = args.mode # 'soft', 'hard', 'auto'

        self.arcface_handler = Arcface_Handler()
        self.projector = Projection_Handler(args)

    def render_uv_image(self, generated, tcoords):
        uv_tmesh = TexturedTriMesh(self.uv_mesh, tcoords, generated, trilist=self.uv_trilist)
        bcs = rasterize_barycentric_coordinate_images(uv_tmesh, self.uv_shape)
        img = rasterize_mesh_from_barycentric_coordinate_images(uv_tmesh, *bcs)
        img.pixels = np.clip(img.pixels, 0.0, 1.0)
        return img

    def render_colored_image(self, view_angle_trg, return_visibility=False):
        uv_cmesh = ColouredTriMesh(self.uv_mesh, trilist=self.uv_trilist, colours=np.tile(view_angle_trg, [3, 1]).T)
        bcs = rasterize_barycentric_coordinate_images(uv_cmesh, self.uv_shape)
        img = rasterize_mesh_from_barycentric_coordinate_images(uv_cmesh, *bcs)
        img.pixels = np.clip(img.pixels, -1.0, 1.0)
        if return_visibility:
            visible = np.sum(bcs[0].pixels, axis=0) != 0
            return img, visible
        else:
            return img

    def camera_tri_angle_src(self, tmesh):#, pose_angle_deg=[0, 0, 0], cam_dist=-4.5):
        camera_direction = -tmesh.points / np.tile(np.linalg.norm(tmesh.points, axis=1), [3, 1]).T
        view_angle = np.sum(camera_direction * tmesh.vertex_normals(), 1)
        return view_angle

    def camera_tri_angle(self, tmesh, pose_angle_deg=[0, 0, 0], cam_dist=-4.5):
        rot_z = rotation_z(pose_angle_deg[2])
        rot_y = rotation_y(-pose_angle_deg[1])
        rot_x = rotation_x(pose_angle_deg[0])
        rotation = rot_z.compose_before(rot_y).compose_before(rot_x)

        translation = Translation([0, 0, +cam_dist])
        camera = rotation.compose_before(translation)

        cam_mesh = camera.apply(tmesh)
        camera_direction = -cam_mesh.points / np.tile(np.linalg.norm(cam_mesh.points, axis=1), [3, 1]).T
        view_angle = np.sum(camera_direction * cam_mesh.vertex_normals(), 1)
        return view_angle

    def create_syn(self, face, trg_angle=[0, 0, 0], include_mask=None):
        view_angle_trg = self.camera_tri_angle(face.tmesh, pose_angle_deg=trg_angle)

        im, projected_mesh = rasterize_image(face.tmesh, self.img_shape, pose_angle_deg=trg_angle, cam_dist=4.5)

        # fill_mask = include_mask.astype(np.bool)
        fill_mask = ((view_angle_trg < face.view_angle_src) | (face.view_angle_src > 0.4)) & self.tight_mask
        if include_mask is not None:
            fill_mask = fill_mask | include_mask.astype(np.bool)
        if face.exclude_mask is not None:
            tcoord_sampling = np.round(self.tcoords.points[:,::-1] * face.exclude_mask.shape).astype(np.int)
            fill_mask[self.mask] =  fill_mask[self.mask] & ~face.exclude_mask[face.exclude_mask.shape[0] - tcoord_sampling[:, 0], tcoord_sampling[:, 1]]

        mask_mesh = ColouredTriMesh(face.tmesh.points, trilist=face.tmesh.trilist, colours=np.tile(fill_mask, [3, 1]).T)
        mask = rasterize_image(mask_mesh, self.img_shape,pose_angle_deg=trg_angle, cam_dist=4.5)[0]

        return im, projected_mesh[:, :2], mask


    def create_align_syn(self, face, trg_angle=[0, 0, 0], include_mask=None):
        im, projected_mesh, mask = self.create_syn(face, trg_angle, include_mask)
        imgs, masks, transformation_params = align_im2stylegan(im_menpo2PIL(im), #im_menpo2PIL(mask),
                                                               im_menpo2PIL(self.extend_mask(im, mask)),
                                                               projected_mesh[self.lms_ind][:,::-1])
        aligned_meshes = align_mesh2stylegan(projected_mesh, transformation_params)
        landmarks = aligned_meshes[self.lms_ind]
        landmarks[:,1] = 1 - landmarks[:,1]

        heatmaps = generate_heatmap.generate_heatmaps(width=self.args.model_res,
                                     height=self.args.model_res,
                                     points=landmarks*self.args.model_res,
                                     sigma=25)

        landmarks = landmarks[:,::-1]
        aligned_meshes = aligned_meshes[self.mask]

        return imgs, masks, heatmaps, aligned_meshes

    def get_tmesh(self, im, reconstruction_dict, face_mask):
        id_features = self.arcface_handler.get_identity_features(im, reconstruction_dict['dense_lms'][self.lms_ind])

        _, yaw_angle, _ = reconstruction_dict['euler_angles']
        is_profile = np.abs(yaw_angle* 180 / np.pi)>30
        visibility_threshold = 0.4
        dense_lms = reconstruction_dict['dense_lms'] / im.shape[::-1]
        dense_lms[:, 1] = 1 - dense_lms[:, 1]

        im_masked = np.array(im_menpo2PIL(im))
        mask_landmarks = np.ones_like(im_masked[:,:,0])
        if face_mask is not None:
            im_masked = im_masked * np.repeat(np.expand_dims(np.array(face_mask,np.bool),2),3,2)
            mask_landmarks *= np.array(face_mask, np.uint8)

        im_masked = fill_UV(im_PIL2menpo(im_masked))
        im_masked.pixels = np.concatenate([im_masked.pixels, np.expand_dims(mask_landmarks,0)],0)
        img_uv_src = self.render_uv_image(im_masked, dense_lms[self.mask])
        mask_landmarks = img_uv_src.pixels[3]<0.5
        img_uv_src.pixels = img_uv_src.pixels[0:3]
        img_uv_src = fill_UV(img_uv_src)
        if is_profile:
            mask_landmarks = binary_dilation(mask_landmarks,disk(5))
            visibility_threshold = 0.6
        img_uv_src.pixels[:,mask_landmarks] = 0
        tcoords = self.tcoords_full.copy()
        tcoords.points[self.mask] = self.tcoords.points

        tmesh = TexturedTriMesh(reconstruction_dict['vertices'], tcoords.points, img_uv_src,
                                trilist=reconstruction_dict['trilist'])
        tmesh_masked = tmesh.from_mask(self.mask)
        tmesh_rotated = TexturedTriMesh(reconstruction_dict['vertices_rotated'], tmesh.tcoords.points, tmesh.texture,
                                trilist=tmesh.trilist)
        view_angle_src = self.camera_tri_angle_src(tmesh_rotated)
        view_angle_src_masked = view_angle_src[self.mask]
        view_angle_src_masked[~self.tight_mask[self.mask]] = -1 # Only take tight crop from the original image
        angle_uv_src, visible = self.render_colored_image(view_angle_src_masked, return_visibility=True)
        angle_uv_src.pixels[:,~visible | mask_landmarks] = -1.0

        mask = angle_uv_src.pixels[0] < visibility_threshold
        mask = ~remove_small_holes(~mask, area_threshold=100000)

        if is_profile and self.mode=='soft':
            mask = binary_dilation(mask, disk(10))
            img_uv_src.pixels[:, mask] = 0
            angle_uv_src.pixels[:, mask] = -1

            img_uv_src_flipped = img_uv_src.mirror(1)
            angle_uv_src_flipped = angle_uv_src.mirror(1)
            temp = img_uv_src_flipped.pixels
            pad = int((16 / 1024) * self.uv_shape[1])
            img_uv_src_flipped.pixels = np.concatenate(
                [np.zeros([temp.shape[0], temp.shape[1], pad]), temp[:, :, :-pad]], 2)
            temp = angle_uv_src_flipped.pixels
            angle_uv_src_flipped.pixels = np.concatenate(
                [np.zeros([temp.shape[0], temp.shape[1], pad]), temp[:, :, :-pad]], 2)


            img_uv_src_flipped = fill_UV(img_uv_src_flipped)
            img_uv_src = fill_UV(img_uv_src)
            mask_flipped = (angle_uv_src_flipped.pixels[0] > visibility_threshold) & mask
            mask_flipped = remove_small_holes(mask_flipped, area_threshold=100000)
            # mask_flipped = binary_dilation(mask_flipped, disk(15))
            angle_uv_src.pixels = mask_flipped * angle_uv_src_flipped.pixels + (1 - mask_flipped) * angle_uv_src.pixels

            mask_all = mask_flipped.astype(int).copy()
            mask_all[~mask_flipped & ~mask] = 2
            mask_all = fill_UV(Image(np.tile(mask_all, [3, 1, 1]))).pixels[0]

            mask_flipped_g = gaussian(mask_all == 1, sigma=30, multichannel=True, mode='reflect')
            mask_flipped_inv_g = gaussian(mask_all == 2, sigma=30, multichannel=True, mode='reflect')

            img_uv_src.pixels = mask_flipped_g * img_uv_src_flipped.pixels + mask_flipped_inv_g * img_uv_src.pixels
            # img_uv_src.pixels[:,mask_flipped] = img_uv_src_flipped.pixels[:,mask_flipped]

            mask = (angle_uv_src.pixels[0] < visibility_threshold)
            mask = ~remove_small_holes(~mask, area_threshold=100000)
            # mask = binary_dilation(mask, disk(15))
            img_uv_src.pixels[:, mask] = 0
            angle_uv_src.pixels[:, mask] = -1

        img_uv_src = fill_UV(img_uv_src)
        tmesh.texture = img_uv_src
        tmesh_rotated.texture = img_uv_src

        face = Face(
            tmesh=tmesh,
            tmesh_masked=tmesh_masked,
            tmesh_rotated=tmesh_rotated,
            img_uv_src=img_uv_src,
            angle_uv_src=angle_uv_src,
            angle_uv_list=[],
            img_uv_list=[],
            view_angle_src=view_angle_src,
            id_features=id_features,
            exclude_mask=mask,
            is_profile=is_profile,
            mode=self.mode
        )

        face.angle_uv_list = [np.clip(angle_uv_src.pixels * face.coef_dict()['src'],-1,1)]
        face.img_uv_list = [fill_UV(img_uv_src).pixels]

        return face

    def extend_mask(self, im, mask):
        # closed_mask = binary_dilation(mask.pixels[0].astype(np.bool), disk(10))
        # extended_mask = ((np.sum(im.pixels, 0) == 0) & (closed_mask & ~mask.pixels[0].astype(np.bool))) | mask.pixels[0].astype(np.bool)
        # im_filled = remove_small_holes(np.sum(im.pixels, 0) > 0, area_threshold=1000)
        # border = binary_dilation(im_filled, disk(10)) & ~binary_erosion(im_filled, disk(10))
        return mask #Image(extended_mask)# | border)

    def run_iteration(self, face, key, trg_angle):

        imgs, masks, heatmaps, aligned_meshes = self.create_align_syn(face, trg_angle, face.uv_blending[key])

        # Run Optimizer
        generated_imgs, generated_dlatents = self.projector.run_projection({key: imgs},
                                                                           {key: masks},
                                                                           {key: heatmaps},
                                                                           face.id_features)

        img_uv = self.render_uv_image(im_PIL2menpo(generated_imgs[key]), aligned_meshes)
        img_uv =fill_UV(img_uv)
        img_uv = uv_color_normalize(face.img_uv_src, face.angle_uv_src, img_uv, Image(face.angle_uv_list[len(face.img_uv_list)]))

        face.img_uv_list.append(img_uv.pixels)

        final_uv, _ = uv_stiching(face.img_uv_list, face.angle_uv_list[:len(face.img_uv_list)], 40)
        results_dict = {
            'generated_imgs': generated_imgs[key],
            'generated_dlatents': generated_dlatents[key],
            'imgs': imgs,
            'masks': masks,
            'aligned_meshes': aligned_meshes,
            'img_uv': img_uv,
            'final_uv': final_uv
        }

        face.img_uv_src = final_uv
        face.tmesh.texture = final_uv
        face.tmesh.tcoords = self.tcoords_full.copy()
        face.tmesh.tcoords.points[self.mask] = self.tcoords.points

        return face, results_dict

    def run(self, im, reconstruction_dict, face_mask=None):
        start = time.time()
        print('Preprocessing...', end=" ")
        # GANFit compatibility
        if not 'vertices' in reconstruction_dict: # GANFit
            reconstruction_dict['vertices'] = reconstruction_dict['tmesh'].points
            reconstruction_dict['trilist'] = reconstruction_dict['tmesh'].trilist
        if not 'vertices_rotated' in reconstruction_dict: # GANFit
            reconstruction_dict['vertices_rotated'] = apply_camera_only3d(reconstruction_dict['vertices'], reconstruction_dict['camera_params'])
        if not 'euler_angles' in reconstruction_dict: # GANFit
            reconstruction_dict['euler_angles'] =  get_pose(reconstruction_dict['camera_params'])

        # Prepare Textured Trimesh with visible part of the face
        face = self.get_tmesh(im, reconstruction_dict, face_mask)

        img_uv_src = face.img_uv_src.copy()
        angle_uv_src = face.angle_uv_src.copy()
        print('Done in %.2f secs' % (time.time() - start))

        # Prepare view angle maps
        start = time.time()
        print('Building a Visibility Index...', end=" ")
        angle_uv = {}
        key_list = ['src']
        angle_uv_list = [np.clip(angle_uv_src.pixels * face.coef_dict()['src'],-1,1)]

        view_angle_src_full = self.camera_tri_angle_src(face.tmesh_rotated)
        tcoord_sampling = np.round(self.tcoords.points*angle_uv_src.shape).astype(np.int)
        view_angle_src_full[self.mask] = angle_uv_src.pixels[0, angle_uv_src.shape[0] - tcoord_sampling[:, 1], tcoord_sampling[:, 0]]
        view_angle_src_full[~self.tight_mask] = -1  # Only take tight crop from the original image

        angle_list = [np.clip(view_angle_src_full * face.coef_dict()['src'],-1,1)]
        dummy_im = im_menpo2PIL(img_uv_src)

        # For each view calculate angles towards the camera (Visibility scores)
        for key, trg_angle in face.rotation_dict().items():
            view_angle_trg = self.camera_tri_angle(face.tmesh, pose_angle_deg=trg_angle)
            view_angle_trg = np.clip(view_angle_trg * face.coef_dict()[key],-1,1)
            _, projected_mesh = rasterize_image(face.tmesh, self.img_shape, pose_angle_deg=trg_angle,
                                                 cam_dist=4.5)
            _, _, transformation_params = align_im2stylegan(dummy_im, dummy_im,
                                                                             projected_mesh[self.lms_ind, :2][:, ::-1])
            aligned_meshes = align_mesh2stylegan(projected_mesh[:, :2], transformation_params)
            out_of_plane = ((aligned_meshes[:, 0] > 1) |
                            (aligned_meshes[:, 1] > 1) |
                            (aligned_meshes[:, 0] < 0) |
                            (aligned_meshes[:, 1] < 0))

            view_angle_trg[out_of_plane] = -1
            angle_list.append(view_angle_trg)
            angle_uv[key] = self.render_colored_image(view_angle_trg[self.mask])
            angle_uv_list.append(angle_uv[key].pixels)
            key_list.append(key)

        # Building a Visibility Index
        max_ind = np.argmax(angle_list, axis=0)
        max_ind_one_hot = np.zeros((max_ind.size, max_ind.max() + 1))
        max_ind_one_hot[np.arange(max_ind.size), max_ind.flatten()] = 1
        max_ind_one_hot = max_ind_one_hot.reshape(max_ind.shape + (-1,))
        mask_out_all = np.max(angle_list,axis=0) ==-1
        max_ind_one_hot[mask_out_all,:] = 0

        uv_blending = {}
        for i, key in enumerate(key_list):
            uv_blending[key] = np.zeros(max_ind_one_hot[:,i].shape,np.float)
            for j in range(i):
                uv_blending[key] += max_ind_one_hot[:,j]
            uv_blending[key] = np.clip(uv_blending[key],0, 1)

        face.uv_blending = uv_blending
        face.angle_uv_list = angle_uv_list
        print('Done in %.2f secs' % (time.time() - start))

        # Projecting for each of the predefined views
        start = time.time()
        print('Projecting...')
        results_dict = {}
        for key, trg_angle in face.rotation_dict().items():
            face, results_dict[key] = self.run_iteration(face, key, trg_angle)

        final_uv = results_dict[key]['final_uv']
        print('Done in %.2f secs' % (time.time() - start))

        if self.args.frontalize:
            start = time.time()
            print('Frontalizing...')
            imgs = {}
            masks = {}
            heatmaps = {}
            self.projector.perceptual_model.assign_placeholder('id_loss', self.args.use_id_loss_frontalize)

            imgs['frontal'], masks['frontal'], heatmaps['frontal'], _ = self.create_align_syn(face, trg_angle=[0, 0, 0], include_mask=face.uv_blending[key])
            generated_imgs, generated_dlatents = self.projector.run_projection(imgs, masks, heatmaps, face.id_features, iterations= self.args.iterations_frontalize)
            results_dict['frontal'] = im_PIL2menpo(generated_imgs['frontal'])
            results_dict['frontalize'] = [imgs, masks, heatmaps, face.id_features]
            print('Done in %.2f secs' % (time.time() - start))

        return final_uv, results_dict
