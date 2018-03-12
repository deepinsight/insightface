from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
#import facenet
import detect_face
import random
from time import sleep
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image

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
    src_path,_ = os.path.split(os.path.realpath(__file__))
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = face_image.get_dataset(args.name, args.input_dir)
    print('dataset size', args.name, len(dataset))
    
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
    if args.name=='lfw' or args.name=='webface' or args.name=='vgg':
      minsize = 20
      threshold = [0.6,0.7,0.9]
      factor = 0.85
    if args.name=='ytf':
      minsize = 20
      threshold = [0.6,0.7,0.4]
      factor = 0.85

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
        for fimage in dataset:
            if nrof_images_total%100==0:
              print("Processing %d, (%d)" % (nrof_images_total, nrof_successfully_aligned))
            nrof_images_total += 1
            image_path = fimage.image_path
            if not os.path.exists(image_path):
              print('image not found (%s)'%image_path)
              continue
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            #print(image_path)
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
                _minsize = minsize
                if fimage.bbox is not None:
                  _bb = fimage.bbox
                  _minsize = min( [_bb[2]-_bb[0], _bb[3]-_bb[1], img.shape[0]//2, img.shape[1]//2] )

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
                  #if bindex<0:
                  #  _img = img[fimage.bbox[1]:fimage.bbox[3], fimage.bbox[0]:fimage.bbox[2],:]
                  #  woffset = fimage.bbox[0]
                  #  hoffset = fimage.bbox[1]
                  #  _minsize = min( [_img.shape[0]//3, _img.shape[1]//3] )
                  #  bounding_boxes, points = detect_face.detect_face(_img, _minsize, pnet, rnet, onet, [0.6,0.7,0.01], factor)
                  #  nrof_faces = bounding_boxes.shape[0]
                  #  print(nrof_faces)
                  #  if nrof_faces>0:
                  #    #print(points.shape)
                  #    #assert(nrof_faces>0)
                  #    bounding_boxes[:,0]+=woffset
                  #    bounding_boxes[:,2]+=woffset
                  #    bounding_boxes[:,1]+=hoffset
                  #    bounding_boxes[:,3]+=hoffset
                  #    points[0:5,:] += woffset
                  #    points[5:10,:] += hoffset
                  #    bindex = 0
                  #    score = bounding_boxes[bindex,4]
                  #    print(score)
                  #    if score<=0.0:
                  #      bindex = -1
                  #    else:
                  #      nrof_force+=1
                  #if bindex<0:
                  #  _bb = fimage.bbox
                  #  _minsize = min( [_bb[2]-_bb[0], _bb[3]-_bb[1], img.shape[0]//2, img.shape[1]//2] )
                  #  bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, [0.6,0.7,0.1], factor)
                  #  nrof_faces = bounding_boxes.shape[0]
                  #  print(nrof_faces)
                  #  if nrof_faces>0:
                  #    bindex = 0
                #if fimage.bbox is not None and bounding_boxes.shape[0]==0:
                #  bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, [0.6,0.7,0.3], factor)


                #print(bounding_boxes.shape, points.shape)
                #print(nrof_faces, points.shape)
                        
                if bindex>=0:

                    det = bounding_boxes[:,0:4]
                    det = det[bindex,:]
                    points = points[:, bindex]
                    #points need to be transpose, points = points.reshape( (5,2) ).transpose()
                    det = np.squeeze(det)
                    #bb = np.zeros(4, dtype=np.int32)
                    #bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    #bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    #bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    #bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                    bb = det
                    #print(points.shape)
                    points = list(points.flatten())
                    assert(len(points)==10)
                    #cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    #scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                    #misc.imsave(output_filename, scaled)
                    nrof_successfully_aligned += 1
                    oline = '%d\t%s\t%d\t%d\t%d\t%d\t%d\t' % (0,fimage.image_path, int(fimage.classname), bb[0], bb[1], bb[2], bb[3])
                    oline += '\t'.join([str(x) for x in points])
                    text_file.write("%s\n"%oline)
                else:
                    print('Unable to align "%s", no face detected' % image_path)
                    if args.force>0:
                      if fimage.bbox is None:
                        oline = '%d\t%s\t%d\n' % (0,fimage.image_path, int(fimage.classname))
                      else:
                        bb = fimage.bbox
                        oline = '%d\t%s\t%d\t%d\t%d\t%d\t%d\n' % (0,fimage.image_path, int(fimage.classname), bb[0], bb[1], bb[2], bb[3])
                      text_file.write(oline)
                      #text_file.write('%s\n' % (output_filename))
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    print('Number of changed: %d' % nrof_changed)
    print('Number of iou3: %d' % nrof_iou3)
    print('Number of force: %d' % nrof_force)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('--name', type=str, help='dataset name, can be facescrub, megaface, webface, celeb.')
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--force', type=int, help='force to output if no faces detected.', default=1)
    #parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
