# -*- coding: utf-8 -*-
#aligned_images_DB 
#--john
#----12.jpg
#----13.jpg

import sys
import os
from easydict import EasyDict as edict

input_dir = '/raid5data/dplearn/YTF/aligned_images_DB'
fp = open('prefix.lst', 'w')
ret = []
label = 0
person_names = []
for person_name in os.listdir(input_dir):
  person_names.append(person_name)
person_names = sorted(person_names)
for person_name in person_names:
  _subdir = os.path.join(input_dir, person_name)
  if not os.path.isdir(_subdir):
    continue
  #for _subdir2 in os.listdir(_subdir):
    #_subdir2 = os.path.join(_subdir, _subdir2)
    #if not os.path.isdir(_subdir2):
    #  continue
  _ret = []
  for img in os.listdir(_subdir):
      fimage = edict()
      fimage.id = os.path.join(_subdir, img)
      fimage.classname = str(label)
      fimage.image_path = os.path.join(_subdir, img)
      fimage.bbox = None
      fimage.landmark = None
      _ret.append(fimage)
  ret += _ret
  label+=1
for item in ret:
  print("%d\t%s\t%d" % (1, item.image_path, int(item.classname)))
  fp.write("%d\t%s\t%d\n" % (1, item.image_path, int(item.classname)))
