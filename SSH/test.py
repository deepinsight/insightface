import cv2
import sys
import numpy as np
import datetime
#sys.path.append('.')
from ssh_detector import SSHDetector

scales = [1200, 1600]
#scales = [600, 1200]
t = 2
detector = SSHDetector('./model/e2ef', 0)


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
  faces = detector.detect(img)
timea = datetime.datetime.now()
faces = detector.detect(img, threshold=0.5)
timeb = datetime.datetime.now()
diff = timeb - timea
print('detection uses', diff.total_seconds(), 'seconds')
print('find', faces.shape[0], 'faces')
