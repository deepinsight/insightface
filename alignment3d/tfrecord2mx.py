
import tensorflow as tf
import cv2
import sys
import mxnet as mx
import os
import numpy as np

input_dir = sys.argv[1]
output_dir = './data'
writer = mx.recordio.MXIndexedRecordIO(os.path.join(output_dir, 'train.idx'), os.path.join(output_dir, 'train.rec'), 'w')

idx = 1
for _file in os.listdir(input_dir):
  if not _file.endswith('tfrecords'):
    continue
  data_file = os.path.join(input_dir, _file)
  for serialized_example in tf.python_io.tf_record_iterator(data_file):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    features = example.features.feature
    image = features['image'].bytes_list.value[0]
    width = features['width'].int64_list.value[0]
    height = features['height'].int64_list.value[0]
    image = np.fromstring(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_COLOR)
    #print(image.shape)
    n_landmarks = features['n_landmarks'].int64_list.value[0]
    mask_index = features['mask_index'].bytes_list.value[0]
    status = features['status'].int64_list.value[0]
    gt_mask = features['gt_mask'].bytes_list.value[0]
    gt_mask = np.fromstring(gt_mask, dtype=np.uint8)
    #print(gt_mask.shape)
    gt_pts = features['gt_pts'].bytes_list.value[0]
    gt_pts = np.fromstring(gt_pts, dtype=np.float32)
    #print(gt_pts.shape, n_landmarks)
    #print(gt_pts)
    #for k in features:
    #  print(k)

    #print(len(image),width, height, n_landmarks, status, gt_mask, gt_pts)
    nlabel = list(gt_pts)
    nheader = mx.recordio.IRHeader(0, nlabel, idx, 0)
    s = mx.recordio.pack_img(nheader, image, quality=95, img_fmt='.jpg')
    writer.write_idx(idx, s)
    idx+=1

