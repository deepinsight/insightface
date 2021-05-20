import os, sys, datetime
import numpy as np
import os.path as osp
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel
import cv2

class MaskRenderer:
    def __init__(self,model_dir, render_only=False):
        self.bfm = MorphabelModel(osp.join(model_dir, 'BFM.mat'))
        self.index_ind = self.bfm.kpt_ind
        uv_coords = face3d.morphable_model.load.load_uv_coords(osp.join(model_dir, 'BFM_UV.mat'))
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
        if not render_only:
            from image_3d68 import Handler
            self.if3d68_handler = Handler(osp.join(model_dir, 'if1k3d68'), 0, 192, ctx_id=0)


    def transform(self, shape3D, R):
        s = 1.0
        shape3D[:2, :] = shape3D[:2, :]
        shape3D = s * np.dot(R, shape3D)
        return shape3D

    def preprocess(self, vertices, w, h):
        R1 = mesh.transform.angle2matrix([0, 180, 180])
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
        image_vertices = mesh.transform.to_image(transformed_vertices, H, W)
        return image_vertices

    def build_params(self, face_image):

        landmark = self.if3d68_handler.get(face_image)[:,:2]
        #print(landmark.shape, landmark.dtype)
        if landmark is None:
            return None #face not found
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

    def render_mask(self,face_image, mask_image, params, auto_blend = True, positions=[0.1, 0.33, 0.9, 0.7]):
        uv_mask_image = self.generate_mask_uv(mask_image, positions)
        h,w,c = face_image.shape
        image_vertices = self.params_to_vertices(params ,h,w)
        output = (1-mesh.render.render_texture(image_vertices, self.bfm.full_triangles , uv_mask_image, self.texcoord, self.bfm.full_triangles, h , w ))*255
        output = output.astype(np.uint8)
        if auto_blend:
            mask_bd = (output==255).astype(np.uint8)
            final = face_image*mask_bd + (1-mask_bd)*output
            return final
        return output


if __name__ == "__main__":
    tool = MaskRenderer('./assets_mask')
    image = cv2.imread("../../deploy/Tom_Hanks_54745.png")
    #mask_image  = cv2.imread("masks/mask1.jpg")
    #mask_image  = cv2.imread("masks/black-mask.png")
    mask_image  = cv2.imread("masks/mask2.jpg")
    params = tool.build_params(image)
    mask_out = tool.render_mask(image, mask_image, params)# use single thread to test the time cost

    cv2.imwrite('output_mask.jpg', mask_out)


