import tensorflow as tf
import numpy as np

from menpo.transform import Translation

from external.landmark_detector.flags import FLAGS

def augment_img(img, augmentation):
    flip, rotate, rescale = np.array(augmentation).squeeze()
    rimg = img.rescale(rescale)
    rimg = rimg.rotate_ccw_about_centre(rotate)
    crimg = rimg.warp_to_shape(
        img.shape,
        Translation(-np.array(img.shape) / 2 + np.array(rimg.shape) / 2)
    )
    if flip > 0.5:
        crimg = crimg.mirror()

    img = crimg

    return img

def rotate_points_tensor(points, image, angle):

    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # center coordinates since rotation center is supposed to be in the image center
    points_centered = points - image_center

    rot_matrix = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)])
    rot_matrix = tf.reshape(rot_matrix, shape=[2, 2])

    points_centered_rot = tf.matmul(rot_matrix, tf.transpose(points_centered))

    return tf.transpose(points_centered_rot) + image_center

def rotate_image_tensor(image, angle):
    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # Coordinates of new image
    xs, ys = tf.meshgrid(tf.range(0.,tf.to_float(s[1])), tf.range(0., tf.to_float(s[0])))
    coords_new = tf.reshape(tf.stack([ys,xs], 2), [-1, 2])

    # center coordinates since rotation center is supposed to be in the image center
    coords_new_centered = tf.to_float(coords_new) - image_center

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.stack(
        [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(
        rot_mat_inv, tf.transpose(coords_new_centered))
    coord_old = tf.to_int32(tf.round(
        tf.transpose(coord_old_centered) + image_center))


    # Find nearest neighbor in old image
    coord_old_y, coord_old_x = tf.unstack(coord_old, axis=1)


    # Clip values to stay inside image coordinates
    outside_y = tf.logical_or(tf.greater(
        coord_old_y, s[0]-1), tf.less(coord_old_y, 0))
    outside_x = tf.logical_or(tf.greater(
        coord_old_x, s[1]-1), tf.less(coord_old_x, 0))
    outside_ind = tf.logical_or(outside_y, outside_x)


    inside_mask = tf.logical_not(outside_ind)
    inside_mask = tf.tile(tf.reshape(inside_mask, s[:2])[...,None], tf.stack([1,1,s[2]]))

    coord_old_y = tf.clip_by_value(coord_old_y, 0, s[0]-1)
    coord_old_x = tf.clip_by_value(coord_old_x, 0, s[1]-1)
    coord_flat = coord_old_y * s[1] + coord_old_x

    im_flat = tf.reshape(image, tf.stack([-1, s[2]]))
    rot_image = tf.gather(im_flat, coord_flat)
    rot_image = tf.reshape(rot_image, s)


    return tf.where(inside_mask, rot_image, tf.zeros_like(rot_image))

def lms_to_heatmap(lms, h, w, n_landmarks, marked_index, sigma=5):
    xs, ys = tf.meshgrid(tf.range(0., tf.to_float(w)),
                         tf.range(0., tf.to_float(h)))
    gaussian = (1. / (sigma * np.sqrt(2. * np.pi)))
    marked_index = tf.to_int32(marked_index)

    def gaussian_fn(lms):
        y, x, idx = tf.unstack(lms)
        idx = tf.to_int32(idx)

        def run_true():
            return tf.exp(-0.5 * (tf.pow(ys - y, 2) + tf.pow(xs - x, 2)) *
                          tf.pow(1. / sigma, 2.)) * gaussian * 17.

        def run_false():
            return tf.zeros((h, w))

        return tf.cond(tf.reduce_any(tf.equal(marked_index, idx)), run_true, run_false)

    img_hm = tf.stack(tf.map_fn(gaussian_fn, tf.concat(
        [lms, tf.to_float(tf.range(0, n_landmarks))[..., None]], 1)))

    return img_hm

class ProtobuffProvider(object):
    def __init__(self, filename= FLAGS['dataset_dir'].value, batch_size=1, rescale=None, augmentation=False):
        self.filename = filename
        self.batch_size = batch_size
        self.image_extension = 'jpg'
        self.rescale = rescale
        self.augmentation = augmentation

    def get(self):
        images, *names = self._get_data_protobuff(self.filename)
        tensors = [images]

        for name in names:
            tensors.append(name)

        return tf.train.shuffle_batch(
            tensors, self.batch_size, 256, 64, self.batch_size)

    def augmentation_type(self):
        return tf.stack([tf.random_uniform([1]) - 1,
                        (tf.random_uniform([1]) * 30. - 15.) * np.pi / 180.,
                        tf.random_uniform([1]) * 0.5 + 0.75])

    def _image_from_feature(self, features):
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])
        image = tf.reshape(image, (image_height, image_width, 3))
        image = tf.to_float(image)
        return image, image_height, image_width

    def _heatmap_from_feature(self, features):
        n_landmarks = tf.to_int32(features['n_landmarks'])
        gt_lms = tf.decode_raw(features['gt_pts'], tf.float32)
        mask_index = tf.decode_raw(features['mask_index'], tf.float32)
        gt_mask = tf.decode_raw(features['gt_mask'], tf.float32)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])

        gt_lms = tf.reshape(gt_lms, (n_landmarks, 2))
        gt_heatmap = lms_to_heatmap(
            gt_lms, image_height, image_width, n_landmarks, mask_index)
        gt_heatmap = tf.transpose(gt_heatmap, perm=[1,2,0])

        return gt_heatmap, gt_lms, n_landmarks, mask_index, gt_mask


    def _info_from_feature(self, features):
        status = features['status']
        return status

    def _set_shape(self, image, gt_heatmap, gt_lms, mask_index, gt_mask):
        image.set_shape([None, None, 3])
        gt_heatmap.set_shape([None, None, FLAGS['n_landmarks']])
        gt_lms.set_shape([FLAGS['n_landmarks'], 2])
        mask_index.set_shape([FLAGS['n_landmarks']])
        gt_mask.set_shape([FLAGS['n_landmarks']])

    def _get_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                # images
                'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                # landmarks
                'n_landmarks': tf.FixedLenFeature([], tf.int64),
                'gt_pts': tf.FixedLenFeature([], tf.string),
                'gt_mask': tf.FixedLenFeature([], tf.string),
                'mask_index': tf.FixedLenFeature([], tf.string),
                'status': tf.FixedLenFeature([], tf.int64),
            }

        )
        return features

    def _get_data_protobuff(self, filename):
        filename = str(filename).split(',')
        filename_queue = tf.train.string_input_producer(filename,
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = self._get_features(serialized_example)

        # image
        image, image_height, image_width = self._image_from_feature(features)

        # landmarks
        gt_heatmap, gt_lms, n_landmarks, mask_index, gt_mask = self._heatmap_from_feature(features)

        # infomations
        status = self._info_from_feature(features)

        # augmentation
        if self.augmentation:
            do_flip, do_rotate, do_scale = tf.unstack(self.augmentation_type())

            # rescale
            image_height = tf.to_int32(tf.to_float(image_height) * do_scale[0])
            image_width = tf.to_int32(tf.to_float(image_width) * do_scale[0])

            image = tf.image.resize_images(image, tf.stack([image_height, image_width]))
            gt_heatmap = tf.image.resize_images(gt_heatmap, tf.stack([image_height, image_width]))
            gt_lms = gt_lms*do_scale

            # rotate
            image = rotate_image_tensor(image, do_rotate)
            gt_heatmap = rotate_image_tensor(gt_heatmap, do_rotate)
            gt_lms = rotate_points_tensor(gt_lms, image, do_rotate)

        # crop to 256 * 256
        target_h = tf.to_int32(256)
        target_w = tf.to_int32(256)
        offset_h = tf.to_int32((image_height - target_h) / 2)
        offset_w = tf.to_int32((image_width - target_w) / 2)

        image = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, target_h, target_w)

        gt_heatmap = tf.image.crop_to_bounding_box(
            gt_heatmap, offset_h, offset_w, target_h, target_w)

        gt_lms = gt_lms - tf.to_float(tf.stack([offset_h, offset_w]))

        self._set_shape(image, gt_heatmap, gt_lms, mask_index, gt_mask)

        return image, gt_heatmap, gt_lms, mask_index, gt_mask
