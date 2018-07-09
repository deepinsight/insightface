import cv2
import sys
import numpy as np
import datetime
#sys.path.append('.')
from ssh_detector import SSHDetector

long_max = 1200
t = 2


f = 't2.jpg'
if len(sys.argv)>1:
  f = sys.argv[1]
img = cv2.imread(f)
print(img.shape)
if img.shape[0]>long_max or img.shape[1]>long_max:
  scale = float(long_max) / max(img.shape[0], img.shape[1])
  img = cv2.resize(img, (0,0), fx=scale, fy=scale)
  print('resize to', img.shape)
detector = SSHDetector('./model/e2ef', 0)
for i in xrange(t-1): #warmup
  faces = detector.detect(img)
timea = datetime.datetime.now()
faces = detector.detect(img)
timeb = datetime.datetime.now()
diff = timeb - timea
print('detection uses', diff.total_seconds(), 'seconds')
print('find', faces.shape[0], 'faces')
