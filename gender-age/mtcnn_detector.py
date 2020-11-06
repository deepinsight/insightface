# coding: utf-8
import os
import mxnet as mx
import numpy as np
import math
import cv2
from multiprocessing import Pool
from itertools import repeat
from itertools import izip
from helper import nms, adjust_input, generate_bbox, detect_first_stage_warpper


class MtcnnDetector(object):
    """
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
        this is a mxnet version
    """
    def __init__(self,
                 model_folder='.',
                 minsize=20,
                 threshold=[0.6, 0.7, 0.8],
                 factor=0.709,
                 num_worker=1,
                 accurate_landmark=False,
                 ctx=mx.cpu()):
        """
            Initialize the detector

            Parameters:
            ----------
                model_folder : string
                    path for the models
                minsize : float number
                    minimal face to detect
                threshold : float number
                    detect threshold for 3 stages
                factor: float number
                    scale factor for image pyramid
                num_worker: int number
                    number of processes we use for first stage
                accurate_landmark: bool
                    use accurate landmark localization or not

        """
        self.num_worker = num_worker
        self.accurate_landmark = accurate_landmark

        # load 4 models from folder
        models = ['det1', 'det2', 'det3', 'det4']
        models = [os.path.join(model_folder, f) for f in models]

        self.PNets = []
        for i in range(num_worker):
            workner_net = mx.model.FeedForward.load(models[0], 1, ctx=ctx)
            self.PNets.append(workner_net)

        #self.Pool = Pool(num_worker)

        self.RNet = mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.ONet = mx.model.FeedForward.load(models[2], 1, ctx=ctx)
        self.LNet = mx.model.FeedForward.load(models[3], 1, ctx=ctx)

        self.minsize = float(minsize)
        self.factor = float(factor)
        self.threshold = threshold

    def convert_to_square(self, bbox):
        """
            convert bbox to square

        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox

        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes

        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxex adjustment

        Returns:
        -------
            bboxes after refinement

        """
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox[:, 0:4] = bbox[:, 0:4] + aug
        return bbox

    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it

        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------s
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox

        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:,
                                                             3] - bboxes[:,
                                                                         1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box, )), np.zeros((num_box, ))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def slice_index(self, number):
        """
            slice the index into (n,n,m), m < n
        Parameters:
        ----------
            number: int number
                number
        """
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        num_list = range(number)
        return list(chunks(num_list, self.num_worker))

    def detect_face_limited(self, img, det_type=2):
        height, width, _ = img.shape
        if det_type >= 2:
            total_boxes = np.array(
                [[0.0, 0.0, img.shape[1], img.shape[0], 0.9]],
                dtype=np.float32)
            num_box = total_boxes.shape[0]

            # pad the bbox
            [dy, edy, dx, edx, y, ey, x, ex, tmpw,
             tmph] = self.pad(total_boxes, width, height)
            # (3, 24, 24) is the input shape for RNet
            input_buf = np.zeros((num_box, 3, 24, 24), dtype=np.float32)

            for i in range(num_box):
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                tmp[dy[i]:edy[i] + 1,
                    dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1,
                                               x[i]:ex[i] + 1, :]
                input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (24, 24)))

            output = self.RNet.predict(input_buf)

            # filter the total_boxes with threshold
            passed = np.where(output[1][:, 1] > self.threshold[1])
            total_boxes = total_boxes[passed]

            if total_boxes.size == 0:
                return None

            total_boxes[:, 4] = output[1][passed, 1].reshape((-1, ))
            reg = output[0][passed]

            # nms
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick]
            total_boxes = self.calibrate_box(total_boxes, reg[pick])
            total_boxes = self.convert_to_square(total_boxes)
            total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])
        else:
            total_boxes = np.array(
                [[0.0, 0.0, img.shape[1], img.shape[0], 0.9]],
                dtype=np.float32)
        num_box = total_boxes.shape[0]
        [dy, edy, dx, edx, y, ey, x, ex, tmpw,
         tmph] = self.pad(total_boxes, width, height)
        # (3, 48, 48) is the input shape for ONet
        input_buf = np.zeros((num_box, 3, 48, 48), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1,
                                                             x[i]:ex[i] + 1, :]
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (48, 48)))

        output = self.ONet.predict(input_buf)
        #print(output[2])

        # filter the total_boxes with threshold
        passed = np.where(output[2][:, 1] > self.threshold[2])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output[2][passed, 1].reshape((-1, ))
        reg = output[1][passed]
        points = output[0][passed]

        # compute landmark points
        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[:, 0:5] = np.expand_dims(
            total_boxes[:, 0], 1) + np.expand_dims(bbw, 1) * points[:, 0:5]
        points[:, 5:10] = np.expand_dims(
            total_boxes[:, 1], 1) + np.expand_dims(bbh, 1) * points[:, 5:10]

        # nms
        total_boxes = self.calibrate_box(total_boxes, reg)
        pick = nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[pick]
        points = points[pick]

        if not self.accurate_landmark:
            return total_boxes, points

        #############################################
        # extended stage
        #############################################
        num_box = total_boxes.shape[0]
        patchw = np.maximum(total_boxes[:, 2] - total_boxes[:, 0] + 1,
                            total_boxes[:, 3] - total_boxes[:, 1] + 1)
        patchw = np.round(patchw * 0.25)

        # make it even
        patchw[np.where(np.mod(patchw, 2) == 1)] += 1

        input_buf = np.zeros((num_box, 15, 24, 24), dtype=np.float32)
        for i in range(5):
            x, y = points[:, i], points[:, i + 5]
            x, y = np.round(x - 0.5 * patchw), np.round(y - 0.5 * patchw)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(
                np.vstack([x, y, x + patchw - 1, y + patchw - 1]).T, width,
                height)
            for j in range(num_box):
                tmpim = np.zeros((tmpw[j], tmpw[j], 3), dtype=np.float32)
                tmpim[dy[j]:edy[j] + 1,
                      dx[j]:edx[j] + 1, :] = img[y[j]:ey[j] + 1,
                                                 x[j]:ex[j] + 1, :]
                input_buf[j, i * 3:i * 3 + 3, :, :] = adjust_input(
                    cv2.resize(tmpim, (24, 24)))

        output = self.LNet.predict(input_buf)

        pointx = np.zeros((num_box, 5))
        pointy = np.zeros((num_box, 5))

        for k in range(5):
            # do not make a large movement
            tmp_index = np.where(np.abs(output[k] - 0.5) > 0.35)
            output[k][tmp_index[0]] = 0.5

            pointx[:, k] = np.round(points[:, k] -
                                    0.5 * patchw) + output[k][:, 0] * patchw
            pointy[:, k] = np.round(points[:, k + 5] -
                                    0.5 * patchw) + output[k][:, 1] * patchw

        points = np.hstack([pointx, pointy])
        points = points.astype(np.int32)

        return total_boxes, points

    def detect_face(self, img, det_type=0):
        """
            detect face over img
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
        Retures:
        -------
            bboxes: numpy array, n x 5 (x1,y2,x2,y2,score)
                bboxes
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
                landmarks
        """

        # check input
        height, width, _ = img.shape
        if det_type == 0:
            MIN_DET_SIZE = 12

            if img is None:
                return None

            # only works for color image
            if len(img.shape) != 3:
                return None

            # detected boxes
            total_boxes = []

            minl = min(height, width)

            # get all the valid scales
            scales = []
            m = MIN_DET_SIZE / self.minsize
            minl *= m
            factor_count = 0
            while minl > MIN_DET_SIZE:
                scales.append(m * self.factor**factor_count)
                minl *= self.factor
                factor_count += 1

            #############################################
            # first stage
            #############################################
            #for scale in scales:
            #    return_boxes = self.detect_first_stage(img, scale, 0)
            #    if return_boxes is not None:
            #        total_boxes.append(return_boxes)

            sliced_index = self.slice_index(len(scales))
            total_boxes = []
            for batch in sliced_index:
                #local_boxes = self.Pool.map( detect_first_stage_warpper, \
                #        izip(repeat(img), self.PNets[:len(batch)], [scales[i] for i in batch], repeat(self.threshold[0])) )
                local_boxes = map( detect_first_stage_warpper, \
                        izip(repeat(img), self.PNets[:len(batch)], [scales[i] for i in batch], repeat(self.threshold[0])) )
                total_boxes.extend(local_boxes)

            # remove the Nones
            total_boxes = [i for i in total_boxes if i is not None]

            if len(total_boxes) == 0:
                return None

            total_boxes = np.vstack(total_boxes)

            if total_boxes.size == 0:
                return None

            # merge the detection from first stage
            pick = nms(total_boxes[:, 0:5], 0.7, 'Union')
            total_boxes = total_boxes[pick]

            bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
            bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1

            # refine the bboxes
            total_boxes = np.vstack([
                total_boxes[:, 0] + total_boxes[:, 5] * bbw,
                total_boxes[:, 1] + total_boxes[:, 6] * bbh,
                total_boxes[:, 2] + total_boxes[:, 7] * bbw,
                total_boxes[:, 3] + total_boxes[:, 8] * bbh, total_boxes[:, 4]
            ])

            total_boxes = total_boxes.T
            total_boxes = self.convert_to_square(total_boxes)
            total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])
        else:
            total_boxes = np.array(
                [[0.0, 0.0, img.shape[1], img.shape[0], 0.9]],
                dtype=np.float32)

        #############################################
        # second stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw,
         tmph] = self.pad(total_boxes, width, height)
        # (3, 24, 24) is the input shape for RNet
        input_buf = np.zeros((num_box, 3, 24, 24), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1,
                                                             x[i]:ex[i] + 1, :]
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (24, 24)))

        output = self.RNet.predict(input_buf)

        # filter the total_boxes with threshold
        passed = np.where(output[1][:, 1] > self.threshold[1])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output[1][passed, 1].reshape((-1, ))
        reg = output[0][passed]

        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick]
        total_boxes = self.calibrate_box(total_boxes, reg[pick])
        total_boxes = self.convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

        #############################################
        # third stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw,
         tmph] = self.pad(total_boxes, width, height)
        # (3, 48, 48) is the input shape for ONet
        input_buf = np.zeros((num_box, 3, 48, 48), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1,
                                                             x[i]:ex[i] + 1, :]
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (48, 48)))

        output = self.ONet.predict(input_buf)

        # filter the total_boxes with threshold
        passed = np.where(output[2][:, 1] > self.threshold[2])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output[2][passed, 1].reshape((-1, ))
        reg = output[1][passed]
        points = output[0][passed]

        # compute landmark points
        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[:, 0:5] = np.expand_dims(
            total_boxes[:, 0], 1) + np.expand_dims(bbw, 1) * points[:, 0:5]
        points[:, 5:10] = np.expand_dims(
            total_boxes[:, 1], 1) + np.expand_dims(bbh, 1) * points[:, 5:10]

        # nms
        total_boxes = self.calibrate_box(total_boxes, reg)
        pick = nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[pick]
        points = points[pick]

        if not self.accurate_landmark:
            return total_boxes, points

        #############################################
        # extended stage
        #############################################
        num_box = total_boxes.shape[0]
        patchw = np.maximum(total_boxes[:, 2] - total_boxes[:, 0] + 1,
                            total_boxes[:, 3] - total_boxes[:, 1] + 1)
        patchw = np.round(patchw * 0.25)

        # make it even
        patchw[np.where(np.mod(patchw, 2) == 1)] += 1

        input_buf = np.zeros((num_box, 15, 24, 24), dtype=np.float32)
        for i in range(5):
            x, y = points[:, i], points[:, i + 5]
            x, y = np.round(x - 0.5 * patchw), np.round(y - 0.5 * patchw)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(
                np.vstack([x, y, x + patchw - 1, y + patchw - 1]).T, width,
                height)
            for j in range(num_box):
                tmpim = np.zeros((tmpw[j], tmpw[j], 3), dtype=np.float32)
                tmpim[dy[j]:edy[j] + 1,
                      dx[j]:edx[j] + 1, :] = img[y[j]:ey[j] + 1,
                                                 x[j]:ex[j] + 1, :]
                input_buf[j, i * 3:i * 3 + 3, :, :] = adjust_input(
                    cv2.resize(tmpim, (24, 24)))

        output = self.LNet.predict(input_buf)

        pointx = np.zeros((num_box, 5))
        pointy = np.zeros((num_box, 5))

        for k in range(5):
            # do not make a large movement
            tmp_index = np.where(np.abs(output[k] - 0.5) > 0.35)
            output[k][tmp_index[0]] = 0.5

            pointx[:, k] = np.round(points[:, k] -
                                    0.5 * patchw) + output[k][:, 0] * patchw
            pointy[:, k] = np.round(points[:, k + 5] -
                                    0.5 * patchw) + output[k][:, 1] * patchw

        points = np.hstack([pointx, pointy])
        points = points.astype(np.int32)

        return total_boxes, points

    def list2colmatrix(self, pts_list):
        """
            convert list to column matrix
        Parameters:
        ----------
            pts_list:
                input list
        Retures:
        -------
            colMat: 

        """
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        colMat = np.matrix(colMat).transpose()
        return colMat

    def find_tfrom_between_shapes(self, from_shape, to_shape):
        """
            find transform between shapes
        Parameters:
        ----------
            from_shape: 
            to_shape: 
        Retures:
        -------
            tran_m:
            tran_b:
        """
        assert from_shape.shape[0] == to_shape.shape[
            0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(from_shape.shape[0] / 2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0] / 2, 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() -
                    mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def extract_image_chips(self, img, points, desired_size=256, padding=0):
        """
            crop and align face
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
            desired_size: default 256
            padding: default 0
        Retures:
        -------
            crop_imgs: list, n
                cropped and aligned faces 
        """
        crop_imgs = []
        for p in points:
            shape = []
            for k in range(len(p) / 2):
                shape.append(p[k])
                shape.append(p[k + 5])

            if padding > 0:
                padding = padding
            else:
                padding = 0
            # average positions of face points
            mean_face_shape_x = [
                0.224152, 0.75610125, 0.490127, 0.254149, 0.726104
            ]
            mean_face_shape_y = [
                0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233
            ]

            from_points = []
            to_points = []

            for i in range(len(shape) / 2):
                x = (padding + mean_face_shape_x[i]) / (2 * padding +
                                                        1) * desired_size
                y = (padding + mean_face_shape_y[i]) / (2 * padding +
                                                        1) * desired_size
                to_points.append([x, y])
                from_points.append([shape[2 * i], shape[2 * i + 1]])

            # convert the points to Mat
            from_mat = self.list2colmatrix(from_points)
            to_mat = self.list2colmatrix(to_points)

            # compute the similar transfrom
            tran_m, tran_b = self.find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
            angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0,
                                                                            0])

            from_center = [(shape[0] + shape[2]) / 2.0,
                           (shape[1] + shape[3]) / 2.0]
            to_center = [0, 0]
            to_center[1] = desired_size * 0.4
            to_center[0] = desired_size * 0.5

            ex = to_center[0] - from_center[0]
            ey = to_center[1] - from_center[1]

            rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]),
                                              -1 * angle, scale)
            rot_mat[0][2] += ex
            rot_mat[1][2] += ey

            chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
            crop_imgs.append(chips)

        return crop_imgs
