import cv2
import sys
import numpy as np
import datetime
sys.path.append('../SSH')
sys.path.append('../alignment')
from ssh_detector import SSHDetector
from alignment import Alignment
from embedding import Embedding

#short_max = 800
scales = [1200, 1600]
t = 2

detector = SSHDetector('../SSH/model/e2ef', 0)
alignment = Alignment('../alignment/model/3d_I5', 12)
embedding = Embedding('./model/model', 0)
out_filename = './out.png'

f = '../sample-images/t1.jpg'
if len(sys.argv)>1:
  f = sys.argv[1]
img = cv2.imread(f)
im_shape = img.shape
print(im_shape)
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
if im_size_min>target_size or im_size_max>max_size:
  im_scale = float(target_size) / float(im_size_min)
  # prevent bigger axis from being more than max_size:
  if np.round(im_scale * im_size_max) > max_size:
      im_scale = float(max_size) / float(im_size_max)
  img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
  print('resize to', img.shape)
for i in xrange(t-1): #warmup
  faces = detector.detect(img, 0.5)
timea = datetime.datetime.now()
faces = detector.detect(img, 0.5)
timeb = datetime.datetime.now()
diff = timeb - timea
print('detection uses', diff.total_seconds(), 'seconds')
print('find', faces.shape[0], 'faces')

for face in faces:
  #print(face)
  cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 1)
  w = face[2] - face[0]
  h = face[3] - face[1]
  wc = int( (face[2]+face[0])/2 )
  hc = int( (face[3]+face[1])/2 )
  size = int(max(w, h)*1.3)
  scale = 100.0/max(w,h)
  M = [ 
        [scale, 0, 64-wc*scale],
        [0, scale, 64-hc*scale],
      ]
  M = np.array(M)
  IM = cv2.invertAffineTransform(M)
  #print(M, IM)
  ebox = cv2.warpAffine(img, M, (128, 128))
  #ebox = cv2.getRectSubPix(img, (size, size), (wc, hc))
  landmark = alignment.get(ebox)
  landmark68 = np.zeros( landmark.shape, dtype=np.float32 )
  #print(landmark.shape)
  for l in range(landmark.shape[0]):
    point = np.ones( (3,), dtype=np.float32)
    point[0:2] = landmark[l]
    point = np.dot(IM, point)
    landmark68[l] = point[0:2]
    pp = (int(point[0]), int(point[1]))
    #print(pp)
    cv2.circle(img, (pp[0], pp[1]), 1, (0, 0, 255), 1)
  feat = embedding.get(img, landmark68)
  print(feat)
print('write to', out_filename)
cv2.imwrite(out_filename, img)
