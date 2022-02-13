import os, sys, datetime
import numpy as np
import os.path as osp
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from .face_analysis import FaceAnalysis
from ..utils import get_model_dir
from ..thirdparty import face3d
from ..data import get_image as ins_get_image
from ..utils import DEFAULT_MP_NAME
import cv2

class MaskRenderer:
    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', insfa=None):
        #if insfa is None, enter render_only mode
        self.mp_name = name
        self.root = root
        self.insfa = insfa
        model_dir = get_model_dir(name, root)
        bfm_file = osp.join(model_dir, 'BFM.mat')
        assert osp.exists(bfm_file), 'should contains BFM.mat in your model directory'
        self.bfm = face3d.morphable_model.MorphabelModel(bfm_file)
        self.index_ind = self.bfm.kpt_ind
        bfm_uv_file = osp.join(model_dir, 'BFM_UV.mat')
        assert osp.exists(bfm_uv_file), 'should contains BFM_UV.mat in your model directory'
        uv_coords = face3d.morphable_model.load.load_uv_coords(bfm_uv_file)
        self.uv_size = (224,224)
        self.mask_stxr =  0.1
        self.mask_styr = 0.33
        self.mask_etxr = 0.9
        self.mask_etyr =  0.7
        self.tex_h , self.tex_w, self.tex_c = self.uv_size[1] , self.uv_size[0],3
        texcoord = np.zeros_like(uv_coords)
        texcoord[:, 0] = uv_coords[:, 0] * (self.tex_h - 1)
        texcoord[:, 1] = uv_coords[:, 1] * (self.tex_w - 1)
        texcoord[:, 1] = self.tex_w - texcoord[:, 1] - 1
        self.texcoord = np.hstack((texcoord, np.zeros((texcoord.shape[0], 1))))
        self.X_ind = self.bfm.kpt_ind
        self.mask_image_names = ['mask_white', 'mask_blue', 'mask_black', 'mask_green']
        self.mask_aug_probs = [0.4, 0.4, 0.1, 0.1]
        #self.mask_images = []
        #self.mask_images_rgb = []
        #for image_name in mask_image_names:
        #    mask_image = ins_get_image(image_name)
        #    self.mask_images.append(mask_image)
        #    mask_image_rgb = mask_image[:,:,::-1]
        #    self.mask_images_rgb.append(mask_image_rgb)


    def prepare(self, ctx_id=0, det_thresh=0.5, det_size=(128, 128)):
        self.pre_ctx_id = ctx_id
        self.pre_det_thresh = det_thresh
        self.pre_det_size = det_size

    def transform(self, shape3D, R):
        s = 1.0
        shape3D[:2, :] = shape3D[:2, :]
        shape3D = s * np.dot(R, shape3D)
        return shape3D

    def preprocess(self, vertices, w, h):
        R1 = face3d.mesh.transform.angle2matrix([0, 180, 180])
        t = np.array([-w // 2, -h // 2, 0])
        vertices = vertices.T
        vertices += t
        vertices = self.transform(vertices.T, R1).T
        return vertices

    def project_to_2d(self,vertices,s,angles,t):
        transformed_vertices = self.bfm.transform(vertices, s, angles, t)
        projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection
        return projected_vertices[self.bfm.kpt_ind, :2]

    def params_to_vertices(self,params  , H , W):
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t  = params
        fitted_vertices = self.bfm.generate_vertices(fitted_sp, fitted_ep)
        transformed_vertices = self.bfm.transform(fitted_vertices, fitted_s, fitted_angles,
                                                  fitted_t)
        transformed_vertices = self.preprocess(transformed_vertices.T, W, H)
        image_vertices = face3d.mesh.transform.to_image(transformed_vertices, H, W)
        return image_vertices

    def draw_lmk(self, face_image):
        faces = self.insfa.get(face_image, max_num=1)
        if len(faces)==0:
            return face_image
        return self.insfa.draw_on(face_image, faces)

    def build_params(self, face_image):
        #landmark = self.if3d68_handler.get(face_image)
        #if landmark is None:
        #    return None #face not found
        if self.insfa is None:
            self.insfa = FaceAnalysis(name=self.mp_name, root=self.root, allowed_modules=['detection', 'landmark_3d_68'])
            self.insfa.prepare(ctx_id=self.pre_ctx_id,  det_thresh=self.pre_det_thresh, det_size=self.pre_det_size)

        faces = self.insfa.get(face_image, max_num=1)
        if len(faces)==0:
            return None
        landmark = faces[0].landmark_3d_68[:,:2]
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = self.bfm.fit(landmark, self.X_ind, max_iter = 3)
        return [fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t]

    def generate_mask_uv(self,mask, positions):
        uv_size = (self.uv_size[1], self.uv_size[0], 3)
        h, w, c = uv_size
        uv = np.zeros(shape=(self.uv_size[1],self.uv_size[0], 3), dtype=np.uint8)
        stxr, styr  = positions[0], positions[1]
        etxr, etyr = positions[2], positions[3]
        stx, sty = int(w * stxr), int(h * styr)
        etx, ety = int(w * etxr), int(h * etyr)
        height = ety - sty
        width = etx - stx
        mask = cv2.resize(mask, (width, height))
        uv[sty:ety, stx:etx] = mask
        return uv

    def render_mask(self,face_image, mask_image, params, input_is_rgb=False, auto_blend = True, positions=[0.1, 0.33, 0.9, 0.7]):
        if isinstance(mask_image, str):
            to_rgb = True if input_is_rgb else False
            mask_image = ins_get_image(mask_image, to_rgb=to_rgb)
        uv_mask_image = self.generate_mask_uv(mask_image, positions)
        h,w,c = face_image.shape
        image_vertices = self.params_to_vertices(params ,h,w)
        output = (1-face3d.mesh.render.render_texture(image_vertices, self.bfm.full_triangles , uv_mask_image, self.texcoord, self.bfm.full_triangles, h , w ))*255
        output = output.astype(np.uint8)
        if auto_blend:
            mask_bd = (output==255).astype(np.uint8)
            final = face_image*mask_bd + (1-mask_bd)*output
            return final
        return output

    #def mask_augmentation(self, face_image, label, input_is_rgb=False, p=0.1):
    #    if np.random.random()<p:
    #        assert isinstance(label, (list, np.ndarray)), 'make sure the rec dataset includes mask params'
    #        assert len(label)==237 or len(lable)==235, 'make sure the rec dataset includes mask params'
    #        if len(label)==237:
    #            if label[1]<0.0: #invalid label for mask aug
    #                return face_image
    #            label = label[2:]
    #        params = self.decode_params(label)
    #        mask_image_name = np.random.choice(self.mask_image_names, p=self.mask_aug_probs)
    #        pos = np.random.uniform(0.33, 0.5)
    #        face_image = self.render_mask(face_image, mask_image_name, params, input_is_rgb=input_is_rgb, positions=[0.1, pos, 0.9, 0.7])
    #    return face_image

    @staticmethod
    def encode_params(params):
        p0 = list(params[0])
        p1 = list(params[1])
        p2 = [float(params[2])]
        p3 = list(params[3])
        p4 = list(params[4])
        return p0+p1+p2+p3+p4

    @staticmethod
    def decode_params(params):
        p0 = params[0:199]
        p0 = np.array(p0, dtype=np.float32).reshape( (-1, 1))
        p1 = params[199:228]
        p1 = np.array(p1, dtype=np.float32).reshape( (-1, 1))
        p2 = params[228]
        p3 = tuple(params[229:232])
        p4 = params[232:235]
        p4 = np.array(p4, dtype=np.float32).reshape( (-1, 1))
        return p0, p1, p2, p3, p4
    
class MaskAugmentation(ImageOnlyTransform):

    def __init__(
            self,
            mask_names=['mask_white', 'mask_blue', 'mask_black', 'mask_green'],
            mask_probs=[0.4,0.4,0.1,0.1],
            h_low = 0.33,
            h_high = 0.35,
            always_apply=False,
            p=1.0,
            ):
        super(MaskAugmentation, self).__init__(always_apply, p)
        self.renderer = MaskRenderer()
        assert len(mask_names)>0
        assert len(mask_names)==len(mask_probs)
        self.mask_names = mask_names
        self.mask_probs = mask_probs
        self.h_low = h_low
        self.h_high = h_high
        #self.hlabel = None


    def apply(self, image, hlabel, mask_name, h_pos, **params):
        #print(params.keys())
        #hlabel = params.get('hlabel')
        assert len(hlabel)==237 or len(hlabel)==235, 'make sure the rec dataset includes mask params'
        if len(hlabel)==237:
            if hlabel[1]<0.0:
                return image
            hlabel = hlabel[2:]
        #print(len(hlabel))
        mask_params = self.renderer.decode_params(hlabel)
        image = self.renderer.render_mask(image, mask_name, mask_params, input_is_rgb=True, positions=[0.1, h_pos, 0.9, 0.7])
        return image

    @property
    def targets_as_params(self):
        return ["image", "hlabel"]

    def get_params_dependent_on_targets(self, params):
        hlabel = params['hlabel']
        mask_name = np.random.choice(self.mask_names, p=self.mask_probs)
        h_pos = np.random.uniform(self.h_low, self.h_high)
        return {'hlabel': hlabel, 'mask_name': mask_name, 'h_pos': h_pos}

    def get_transform_init_args_names(self):
        #return ("hlabel", 'mask_names', 'mask_probs', 'h_low', 'h_high')
        return ('mask_names', 'mask_probs', 'h_low', 'h_high')


if __name__ == "__main__":
    tool = MaskRenderer('antelope')
    tool.prepare(det_size=(128,128))
    image = cv2.imread("Tom_Hanks_54745.png")
    params = tool.build_params(image)
    #out = tool.draw_lmk(image)
    #cv2.imwrite('output_lmk.jpg', out)
    #mask_image  = cv2.imread("masks/mask1.jpg")
    #mask_image  = cv2.imread("masks/black-mask.png")
    #mask_image  = cv2.imread("masks/mask2.jpg")
    mask_out = tool.render_mask(image, 'mask_blue', params)# use single thread to test the time cost

    cv2.imwrite('output_mask.jpg', mask_out)


