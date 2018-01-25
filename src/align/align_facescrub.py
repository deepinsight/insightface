from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import json
import argparse
import tensorflow as tf
import numpy as np
#import facenet
import detect_face
import random
from time import sleep
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image
from skimage import transform as trans
import cv2

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe,GTframe):
  x1 = Reframe[0];
  y1 = Reframe[1];
  width1 = Reframe[2]-Reframe[0];
  height1 = Reframe[3]-Reframe[1];

  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]

  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio


def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    image_dir = os.path.join(args.input_dir, 'facescrub')
    dataset = face_image.get_dataset('facescrub', image_dir)
    print('dataset size', len(dataset))
    bbox = {}
    for label_file in ['facescrub_actors.txt',  'facescrub_actresses.txt']:
      label_file = os.path.join(args.input_dir, label_file)
      pp = 0
      for line in open(label_file, 'r'):
        pp+=1
        if pp==1:
          continue
        vec = line.split("\t")
        key = (vec[0], int(vec[2]))
        value = [int(x) for x in vec[4].split(',')]
        bbox[key] = value
    print('bbox size', len(bbox))

    valid_key = {}
    json_data = open(os.path.join(args.input_dir, 'facescrub_uncropped_features_list.json')).read()
    json_data = json.loads(json_data)['path']
    for _data in json_data:
      key = _data.split('/')[-1]
      pos = key.rfind('.')
      if pos<0:
        print(_data)
      else:
        key = key[0:pos]
      keys = key.split('_')
      #print(key)
      if len(keys)!=2:
        print('err', key, _data)
        continue
      #assert len(keys)==2
      key = (keys[0], int(keys[1]))
      valid_key[key] = 1
      #print(key)
    print('valid keys', len(valid_key))
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 100 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    image_size = [112,96]
    image_size = [112,112]
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0

    # Add a random key to the filename to allow alignment using multiple processes
    #random_key = np.random.randint(0, high=99999)
    #bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    #output_filename = os.path.join(output_dir, 'faceinsight_align_%s.lst' % args.name)
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

    output_filename = os.path.join(args.output_dir, 'lst')
    
    
    with open(output_filename, "w") as text_file:
        nrof_images_total = 0
        nrof = np.zeros( (5,), dtype=np.int32)
        for fimage in dataset:
            if nrof_images_total%100==0:
              print("Processing %d, (%s)" % (nrof_images_total, nrof))
            nrof_images_total += 1
            #if nrof_images_total<950000:
            #  continue
            image_path = fimage.image_path
            if not os.path.exists(image_path):
              print('image not found (%s)'%image_path)
              continue
            #print(image_path)
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            _paths = fimage.image_path.split('/')
            print(fimage.image_path)
            a,b = _paths[-2], _paths[-1]
            pb = b.rfind('.')
            bname = b[0:pb]
            pb = bname.rfind('_')
            body = bname[(pb+1):]
            img_id = int(body)
            key = (a, img_id)
            if not key in valid_key:
              continue
            #print(b, img_id)
            assert key in bbox
            fimage.bbox = bbox[key]
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim<2:
                    print('Unable to align "%s", img dim error' % image_path)
                    #text_file.write('%s\n' % (output_filename))
                    continue
                if img.ndim == 2:
                    img = to_rgb(img)
                img = img[:,:,0:3]
                tb = bname.replace(' ','_')+".png"
                ta = a.replace(' ','_')
                target_dir = os.path.join(args.output_dir, ta)
                if not os.path.exists(target_dir):
                  os.makedirs(target_dir)

                target_file = os.path.join(target_dir, tb)
                warped = None
                if fimage.landmark is not None:
                  dst = fimage.landmark.astype(np.float32)

                  tform = trans.SimilarityTransform()
                  tform.estimate(dst, src[0:3,:]*1.5+image_size[0]*0.25)
                  M = tform.params[0:2,:]
                  warped0 = cv2.warpAffine(img,M,(image_size[1]*2,image_size[0]*2), borderValue = 0.0)
                  _minsize = image_size[0]
                  bounding_boxes, points = detect_face.detect_face(warped0, _minsize, pnet, rnet, onet, threshold, factor)
                  if bounding_boxes.shape[0]>0:
                    bindex = 0
                    det = bounding_boxes[bindex,0:4]
                    #points need to be transpose, points = points.reshape( (5,2) ).transpose()
                    dst = points[:, bindex].reshape( (2,5) ).T
                    tform = trans.SimilarityTransform()
                    tform.estimate(dst, src)
                    M = tform.params[0:2,:]
                    warped = cv2.warpAffine(warped0,M,(image_size[1],image_size[0]), borderValue = 0.0)
                    nrof[0]+=1
                #assert fimage.bbox is not None
                if warped is None and fimage.bbox is not None:
                  _minsize = img.shape[0]//4
                  bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
                  if bounding_boxes.shape[0]>0:
                    det = bounding_boxes[:,0:4]
                    bindex = -1
                    index2 = [0.0, 0]
                    for i in xrange(det.shape[0]):
                      _det = det[i]
                      iou = IOU(fimage.bbox, _det)
                      if iou>index2[0]:
                        index2[0] = iou
                        index2[1] = i
                    if index2[0]>0.3:
                      bindex = index2[1]
                    if bindex>=0:
                      dst = points[:, bindex].reshape( (2,5) ).T
                      tform = trans.SimilarityTransform()
                      tform.estimate(dst, src)
                      M = tform.params[0:2,:]
                      warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
                      nrof[1]+=1
                      #print('1',target_file,index2[0])
                if warped is None and fimage.bbox is not None:
                  bb = fimage.bbox
                  #croped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                  bounding_boxes, points = detect_face.detect_face_force(img, bb, pnet, rnet, onet)
                  assert bounding_boxes.shape[0]==1
                  _box = bounding_boxes[0]
                  if _box[4]>=0.3:
                    dst = points[:, 0].reshape( (2,5) ).T
                    tform = trans.SimilarityTransform()
                    tform.estimate(dst, src)
                    M = tform.params[0:2,:]
                    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
                    nrof[2]+=1
                    #print('2',target_file)

                if warped is None:
                  roi = np.zeros( (4,), dtype=np.int32)
                  roi[0] = int(img.shape[1]*0.06)
                  roi[1] = int(img.shape[0]*0.06)
                  roi[2] = img.shape[1]-roi[0]
                  roi[3] = img.shape[0]-roi[1]
                  if fimage.bbox is not None:
                    bb = fimage.bbox
                    h = bb[3]-bb[1]
                    w = bb[2]-bb[0]
                    x = bb[0]
                    y = bb[1]
                    #roi = np.copy(bb)
                    _w = int( (float(h)/image_size[0])*image_size[1] )
                    x += (w-_w)//2
                    #x = min( max(0,x), img.shape[1] )
                    x = max(0,x)
                    xw = x+_w
                    xw = min(xw, img.shape[1])
                    roi = np.array( (x, y, xw, y+h), dtype=np.int32)
                    nrof[3]+=1
                  else:
                    nrof[4]+=1
                  #print('3',bb,roi,img.shape)
                  #print('3',target_file)
                  warped = img[roi[1]:roi[3],roi[0]:roi[2],:]
                  #print(warped.shape)
                  warped = cv2.resize(warped, (image_size[1], image_size[0]))
                bgr = warped[...,::-1]
                cv2.imwrite(target_file, bgr)
                oline = '%d\t%s\t%d\n' % (1,target_file, int(fimage.classname))
                text_file.write(oline)
                            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.')
    #parser.add_argument('--image_size', type=int,
    #    help='Image size (height, width) in pixels.', default=182)
    #parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

