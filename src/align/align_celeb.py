from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import cv2
import argparse
import tensorflow as tf
import numpy as np
import base64
#import facenet
import detect_face
from easydict import EasyDict as edict
import random
from time import sleep
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image
import face_preprocess

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
    datamap = {}
    pp = 0
    datasize = 0
    verr = 0
    for line in open(args.input_dir+"_clean_list.txt", 'r'):
      pp+=1
      if pp%10000==0:
        print('loading list', pp)
      line = line.strip()[2:]
      if not line.startswith('m.'):
        continue
      vec = line.split('/')
      assert len(vec)==2
      #print(line)
      person = vec[0]
      img = vec[1]
      try:
        img_id = int(img.split('.')[0])
      except ValueError:
        #print('value error', line)
        verr+=1
        continue
      if not person in datamap:
        labelid = len(datamap)
        datamap[person] = [labelid, {img_id : 1}]
      else:
        datamap[person][1][img_id] = 1
      datasize+=1

    print('dataset size', args.name, datasize)
    print('dataset err', verr)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 100 # minimum size of face
    #threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    threshold = [ 0.6, 0.6, 0.3 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print(minsize)
    print(threshold)
    print(factor)

    # Add a random key to the filename to allow alignment using multiple processes
    #random_key = np.random.randint(0, high=99999)
    #bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    output_filename = os.path.join(output_dir, 'faceinsight_align_%s.lst' % args.name)
    
    with open(output_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        nrof_changed = 0
        nrof_iou3 = 0
        nrof_force = 0
        for line in open(args.input_dir, 'r'):
            vec = line.strip().split()
            person = vec[0]
            img_id = int(vec[1])
            v = datamap.get(person, None)
            if v is None:
              continue
            #TODO
            #if not img_id in v[1]:
            #  continue
            labelid = v[0]
            img_str = base64.b64decode(vec[-1])
            nparr = np.fromstring(img_str, np.uint8)
            img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
            img = img[...,::-1] #to rgb
            if nrof_images_total%100==0:
              print("Processing %d, (%d)" % (nrof_images_total, nrof_successfully_aligned))
            nrof_images_total += 1
            target_dir = os.path.join(output_dir, person)
            if not os.path.exists(target_dir):
              os.makedirs(target_dir)
            target_path = os.path.join(target_dir, "%d.jpg"%img_id)
            _minsize = minsize
            fimage = edict()
            fimage.bbox = None
            fimage.image_path = target_path
            fimage.classname = str(labelid)
            if fimage.bbox is not None:
              _bb = fimage.bbox
              _minsize = min( [_bb[2]-_bb[0], _bb[3]-_bb[1], img.shape[0]//2, img.shape[1]//2] )
            else:
              _minsize = min(img.shape[0]//5, img.shape[1]//5)
            bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
            bindex = -1
            nrof_faces = bounding_boxes.shape[0]
            if fimage.bbox is None and nrof_faces>0:
              det = bounding_boxes[:,0:4]
              img_size = np.asarray(img.shape)[0:2]
              bindex = 0
              if nrof_faces>1:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
            if fimage.bbox is not None:
              if nrof_faces>0:
                assert(bounding_boxes.shape[0]==points.shape[1])
                det = bounding_boxes[:,0:4]
                img_size = np.asarray(img.shape)[0:2]
                index2 = [0.0, 0]
                for i in xrange(det.shape[0]):
                  _det = det[i]
                  iou = IOU(fimage.bbox, _det)
                  if iou>index2[0]:
                    index2[0] = iou
                    index2[1] = i
                if index2[0]>-0.3:
                  bindex = index2[1]
                  nrof_iou3+=1
              if bindex<0:
                bounding_boxes, points = detect_face.detect_face_force(img, fimage.bbox, pnet, rnet, onet)
                bindex = 0
                nrof_force+=1
                    
            if bindex>=0:

                det = bounding_boxes[:,0:4]
                det = det[bindex,:]
                points = points[:, bindex]
                landmark = points.reshape((2,5)).T
                #points need to be transpose, points = points.reshape( (5,2) ).transpose()
                det = np.squeeze(det)
                bb = det
                points = list(points.flatten())
                assert(len(points)==10)
                warped = face_preprocess.preprocess(img, bbox=bb, landmark = landmark, image_size=args.image_size)
                misc.imsave(target_path, warped)
                nrof_successfully_aligned += 1
                oline = '%d\t%s\t%d' % (1,fimage.image_path, int(fimage.classname))
                #oline = '%d\t%s\t%d\t%d\t%d\t%d\t%d\t' % (0,fimage.image_path, int(fimage.classname), bb[0], bb[1], bb[2], bb[3])
                #oline += '\t'.join([str(x) for x in points])
                text_file.write("%s\n"%oline)
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    print('Number of changed: %d' % nrof_changed)
    print('Number of iou3: %d' % nrof_iou3)
    print('Number of force: %d' % nrof_force)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('--name', type=str, default='celeb', help='')
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image-size', type=str, help='Image size (height, width) in pixels.', default='112,112')
    #parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
