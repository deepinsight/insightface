import cv2
import sys
import numpy as np
import datetime
from alignment import Alignment
sys.path.append('../SSH')
from ssh_detector import SSHDetector

long_max = 1200
t = 2

detector = SSHDetector('../SSH/model/e2ef', 0)
alignment = Alignment('./model/3d_I5', 12)

f = '../sample-images/t2.jpg'
if len(sys.argv)>1:
  f = sys.argv[1]
img = cv2.imread(f)
print(img.shape)
if img.shape[0]>long_max or img.shape[1]>long_max:
  scale = float(long_max) / max(img.shape[0], img.shape[1])
  img = cv2.resize(img, (0,0), fx=scale, fy=scale)
  print('resize to', img.shape)
for i in xrange(t-1): #warmup
  faces = detector.detect(img)
timea = datetime.datetime.now()
faces = detector.detect(img)
timeb = datetime.datetime.now()
diff = timeb - timea
print('detection uses', diff.total_seconds(), 'seconds')
print('find', faces.shape[0], 'faces')

for face in faces:
  w = face[2] - face[0]
  h = face[3] - face[1]
  size = int(max(w, h)*1.3)
  wc = int( (face[2]+face[0])/2 )
  hc = int( (face[3]+face[1])/2 )
  ebox = cv2.getRectSubPix(img, (size, size), (wc, hc))
  landmark = alignment.get(ebox)
  print(landmark.shape)
