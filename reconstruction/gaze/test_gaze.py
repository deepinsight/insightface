
from models import GazeModel
import sys
import glob
import torch
import os
import os.path as osp
import numpy as np
import cv2
import os.path as osp
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import menpo.io as mio
from menpo.image import Image
from menpo.shape import PointCloud, TriMesh


def angles_from_vec(vec):
    x, y, z = -vec[2], vec[1], -vec[0]
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z) - np.pi/2
    theta_x, theta_y = phi, theta
    return theta_x, theta_y

def vec_from_eye(eye, iris_lms_idx):
    p_iris = eye[iris_lms_idx] - eye[:32].mean(axis=0)
    vec = p_iris.mean(axis=0)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def angles_and_vec_from_eye(eye, iris_lms_idx):

    vec = vec_from_eye(eye, iris_lms_idx)
    theta_x, theta_y = angles_from_vec(vec)
    return theta_x, theta_y, vec

def vec_from_angles(rx, ry):
    rx = np.deg2rad(rx)
    ry = np.deg2rad(ry)
    x1 = np.sin(np.pi/2 + rx) * np.cos(ry) 
    y1 = np.sin(np.pi/2 + rx) * np.sin(ry)
    z1 = np.cos(np.pi/2 + rx)
    x, y, z = -z1, y1, -x1
    vec = np.array([x, y, z])
    vec /= np.linalg.norm(vec, axis=0)
    return vec


class GazeHandler():
    def __init__(self, ckpt_path, res_eyes_path='assets/eyes3d.pkl'):
        R = 12.0
        eyes_mean = mio.import_pickle(res_eyes_path)
        idxs481 = eyes_mean['mask481']['idxs']
        self.tri481 = eyes_mean['mask481']['trilist']
        self.iris_idx_481 = eyes_mean['mask481']['idxs_iris']

        self.mean_l = eyes_mean['left_points'][idxs481][:, [0, 2, 1]]
        self.mean_r = eyes_mean['right_points'][idxs481][:, [0, 2, 1]]

        self.num_face = 1103
        self.num_eye = 481
        self.app = FaceAnalysis()
        det_size = 320
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
        self.input_size = 160
        self.model = GazeModel.load_from_checkpoint(ckpt_path).cuda()
        self.model.eval()



    def draw_item(self, eimg, item):
        #bbox, kps, eye_kps = item
        eye_kps = item
        #eye_kps[:,2] *= 10.0
        eye_l = eye_kps[:self.num_eye,:]
        eye_r = eye_kps[self.num_eye:,:]
        for _eye in [eye_l, eye_r]:
            tmp = _eye[:,0].copy()
            _eye[:,0] = _eye[:,1].copy()
            _eye[:,1] = tmp
        #img_crp, trf_crp = image.crop_to_pointcloud_proportion(PointCloud(np.concatenate((eye_l[:, :2], eye_r[:, :2]))), 0.4,  return_transform=True)
        #img_crp.view(1)
        #trf_crp.pseudoinverse().apply(TriMesh(eye_l, tri481).with_dims([0, 1])).view(1, marker_size=0.01, line_width=0.1)
        #trf_crp.pseudoinverse().apply(PointCloud(eye_l[iris_idx_481]).with_dims([0, 1])).view(1)
        #trf_crp.pseudoinverse().apply(TriMesh(eye_r, tri481).with_dims([0, 1])).view(1, marker_size=0.01, line_width=0.1)
        #trf_crp.pseudoinverse().apply(PointCloud(eye_r[iris_idx_481]).with_dims([0, 1])).view(1)

        for _eye in [eye_l, eye_r]:
            _kps = _eye[self.iris_idx_481,:].astype(np.int)
            for l in range(_kps.shape[0]):
                color = (0, 255, 0)
                cv2.circle(eimg, (_kps[l][1], _kps[l][0]), 4, color, 4)
            #print(tri481.shape)
            for _tri in self.tri481:
                color = (0, 0, 255)
                for k in range(3):
                    ix = _tri[k]
                    iy = _tri[(k+1)%3]
                    x = _eye[ix,:2].astype(np.int)[::-1]
                    y = _eye[iy,:2].astype(np.int)[::-1]
                    cv2.line(eimg, x, y, color, 1)

        theta_x_l, theta_y_l, vec_l = angles_and_vec_from_eye(eye_l, self.iris_idx_481)
        theta_x_r, theta_y_r, vec_r = angles_and_vec_from_eye(eye_r, self.iris_idx_481)
        gaze_pred = np.array([(theta_x_l + theta_x_r) / 2, (theta_y_l + theta_y_r) / 2])

        diag = np.sqrt(float(eimg.shape[0]*eimg.shape[1]))

        #img_crp, trf_crp = image.crop_to_pointcloud_proportion(PointCloud(lms[17:-20]), 0.1, return_transform=True)
        #diag = img_crp.diagonal()

        eye_pos_left = eye_l[self.iris_idx_481].mean(axis=0)[[0, 1]]
        eye_pos_right = eye_r[self.iris_idx_481].mean(axis=0)[[0, 1]]

        ##fig = plt.figure(0)
        ##image.view(0, figure_size=(4,4))
        #PointCloud(eye_l[iris_idx_481]).with_dims([0, 1]).view(0, marker_size=3, figure_size=(4,4))
        #PointCloud(eye_r[iris_idx_481]).with_dims([0, 1]).view(0, marker_size=3, figure_size=(4,4))

        ## pred ---
        gaze_pred = np.array([theta_x_l, theta_y_l])
        dx = 0.4*diag * np.sin(gaze_pred[1])
        dy = 0.4*diag * np.sin(gaze_pred[0])
        x = np.array([eye_pos_left[1], eye_pos_left[0]])
        y = x.copy()
        y[0] += dx
        y[1] += dy
        x = x.astype(np.int)
        y = y.astype(np.int)
        color = (0,255,255)
        cv2.line(eimg, x, y, color, 2)

        gaze_pred = np.array([theta_x_r, theta_y_r])
        dx = 0.4*diag * np.sin(gaze_pred[1])
        dy = 0.4*diag * np.sin(gaze_pred[0])
        x = np.array([eye_pos_right[1], eye_pos_right[0]])
        y = x.copy()
        y[0] += dx
        y[1] += dy
        x = x.astype(np.int)
        y = y.astype(np.int)
        color = (0,255,255)
        cv2.line(eimg, x, y, color, 2)
        return eimg

    def draw_on(self, eimg, results):
        face_sizes = [ (x[0][2] - x[0][0]) for x in results]
        max_index = np.argmax(face_sizes)
        max_face_size = face_sizes[max_index]
        rescale = 300.0 / max_face_size
        #print(max_face_size, rescale)
        oimg = eimg.copy()
        eimg = cv2.resize(eimg, None, fx=rescale, fy=rescale)
        for pred in results:
            _, _, eye_kps = pred
            eye_kps = eye_kps.copy()
            eye_kps *= rescale
            eimg = self.draw_item(eimg, eye_kps)
        eimg = cv2.resize(eimg, (oimg.shape[1], oimg.shape[0]))
        return eimg
        pred_max = results[max_index]
        bbox, kps, eye_kps = pred_max
        width = bbox[2] - bbox[0]
        center = (kps[0]+kps[1]) / 2.0
        #_size = np.abs(kps[1][0] - kps[0][0]) * 1.5
        _size = max(width/1.5, np.abs(kps[1][0] - kps[0][0]) ) * 1.5
        rotate = 0
        _scale = self.input_size  / _size
        aimg, M = face_align.transform(oimg, center, self.input_size, _scale, rotate)
        eye_kps = face_align.trans_points(eye_kps, M)
        center_eye_rescale = 4.0
        aimg = cv2.resize(aimg, None, fx=center_eye_rescale, fy=center_eye_rescale)
        eye_kps *= center_eye_rescale
        aimg = self.draw_item(aimg, eye_kps)
        #return aimg
    
        rimg = np.zeros( (max(eimg.shape[0], aimg.shape[0]), eimg.shape[1]+aimg.shape[1], 3), dtype=np.uint8)
        rimg[:eimg.shape[0], :eimg.shape[1], :] = eimg
        rimg[:aimg.shape[0], eimg.shape[1]:eimg.shape[1]+aimg.shape[1], :] = aimg
        return rimg

    def get(self, img):
        results = []
        faces = self.app.get(img)
        if len(faces)==0:
            return results
        for face in faces:
            bbox = face.bbox
            width = bbox[2] - bbox[0]
            kps = face.kps
            center = (kps[0]+kps[1]) / 2.0
            #_size = np.abs(kps[1][0] - kps[0][0]) * 1.5
            _size = max(width/1.5, np.abs(kps[1][0] - kps[0][0]) ) * 1.5
            rotate = 0
            _scale = self.input_size  / _size
            aimg, M = face_align.transform(img, center, self.input_size, _scale, rotate)
            #eimg = cv2.resize(aimg, None, fx=R, fy=R)
            #cv2.imwrite("outputs/a_%s"%name, aimg)
            aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
            input = aimg.copy()
            input = np.transpose(input, (2, 0, 1))
            input = np.expand_dims(input, 0)
            imgs = torch.Tensor(input).cuda()
            imgs.div_(255).sub_(0.5).div_(0.5)
            opred = self.model(imgs).detach().cpu().numpy().flatten().reshape( (-1, 3) )
            opred[:, 0:2] += 1
            opred[:, 0:2] *= (self.input_size // 2)
            #opred[:, 0:2] *= 112
            opred[:,2] *= 10.0
            IM = cv2.invertAffineTransform(M)
            pred = face_align.trans_points(opred, IM)
            result = (bbox, kps, pred)
            results.append(result)
        return results


if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    handler = GazeHandler(ckpt_path)
    output_dir = 'outputs/'
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    input_dir = 'assets/images'
    for imgname in os.listdir(input_dir):
        imgpath = osp.join(input_dir, imgname)
        img = cv2.imread(imgpath)
        print(imgpath, imgname)
        if img is None:
            continue
        results = handler.get(img)
        if len(results)==0:
            continue
        eimg = handler.draw_on(img, results)
        oimg = np.concatenate((img, eimg), axis=1)
        cv2.imwrite(osp.join(output_dir, "%s"%imgname), oimg)



