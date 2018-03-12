
from easydict import EasyDict as edict
import os
import json
import numpy as np


def load_property(data_dir):
  prop = edict()
  for line in open(os.path.join(data_dir, 'property')):
    vec = line.strip().split(',')
    assert len(vec)==3
    prop.num_classes = int(vec[0])
    prop.image_size = [int(vec[1]), int(vec[2])]
  return prop



def get_dataset_webface(input_dir):
  clean_list_file = input_dir+"_clean_list.txt"
  ret = []
  for line in open(clean_list_file, 'r'):
    vec = line.strip().split()
    assert len(vec)==2
    fimage = edict()
    fimage.id = vec[0].replace("\\", '/')
    fimage.classname = vec[1]
    fimage.image_path = os.path.join(input_dir, fimage.id)
    ret.append(fimage)
  return ret

def get_dataset_celeb(input_dir):
  clean_list_file = input_dir+"_clean_list.txt"
  ret = []
  dir2label = {}
  for line in open(clean_list_file, 'r'):
    line = line.strip()
    if not line.startswith('./m.'):
      continue
    line = line[2:]
    vec = line.split('/')
    assert len(vec)==2
    if vec[0] in dir2label:
      label = dir2label[vec[0]]
    else:
      label = len(dir2label)
      dir2label[vec[0]] = label

    fimage = edict()
    fimage.id = line
    fimage.classname = str(label)
    fimage.image_path = os.path.join(input_dir, fimage.id)
    ret.append(fimage)
  return ret

def _get_dataset_celeb(input_dir):
  list_file = input_dir+"_original_list.txt"
  ret = []
  for line in open(list_file, 'r'):
    vec = line.strip().split()
    assert len(vec)==2
    fimage = edict()
    fimage.id = vec[0]
    fimage.classname = vec[1]
    fimage.image_path = os.path.join(input_dir, fimage.id)
    ret.append(fimage)
  return ret

def get_dataset_facescrub(input_dir):
  ret = []
  label = 0
  person_names = []
  for person_name in os.listdir(input_dir):
    person_names.append(person_name)
  person_names = sorted(person_names)
  for person_name in person_names:
    subdir = os.path.join(input_dir, person_name)
    if not os.path.isdir(subdir):
      continue
    for _img in os.listdir(subdir):
      fimage = edict()
      fimage.id = os.path.join(person_name, _img)
      fimage.classname = str(label)
      fimage.image_path = os.path.join(subdir, _img)
      fimage.landmark = None
      fimage.bbox = None
      ret.append(fimage)
    label += 1
  return ret

def get_dataset_megaface(input_dir):
  ret = []
  label = 0
  for prefixdir in os.listdir(input_dir):
    _prefixdir = os.path.join(input_dir, prefixdir)
    for subdir in os.listdir(_prefixdir):
      _subdir = os.path.join(_prefixdir, subdir)
      if not os.path.isdir(_subdir):
        continue
      for img in os.listdir(_subdir):
        if not img.endswith('.jpg.jpg') and img.endswith('.jpg'):
          fimage = edict()
          fimage.id = os.path.join(prefixdir, subdir, img)
          fimage.classname = str(label)
          fimage.image_path = os.path.join(_subdir, img)
          json_file = fimage.image_path+".json"
          data = None
          fimage.bbox = None
          fimage.landmark = None
          if os.path.exists(json_file):
            with open(json_file, 'r') as f:
              data = f.read()
              data = json.loads(data)
            assert data is not None
            if 'bounding_box' in data:
              fimage.bbox = np.zeros( (4,), dtype=np.float32 )
              bb = data['bounding_box']
              fimage.bbox[0] = bb['x']
              fimage.bbox[1] = bb['y']
              fimage.bbox[2] = bb['x']+bb['width']
              fimage.bbox[3] = bb['y']+bb['height']
              #print('bb')
            if 'landmarks' in data:
              landmarks = data['landmarks']
              if '1' in landmarks and '0' in landmarks and '2' in landmarks:
                fimage.landmark = np.zeros( (3,2), dtype=np.float32 )
                fimage.landmark[0][0] = landmarks['1']['x']
                fimage.landmark[0][1] = landmarks['1']['y']
                fimage.landmark[1][0] = landmarks['0']['x']
                fimage.landmark[1][1] = landmarks['0']['y']
                fimage.landmark[2][0] = landmarks['2']['x']
                fimage.landmark[2][1] = landmarks['2']['y']
              #print('lm')

          ret.append(fimage)
      label+=1
  return ret

def get_dataset_fgnet(input_dir):
  ret = []
  label = 0
  for subdir in os.listdir(input_dir):
    _subdir = os.path.join(input_dir, subdir)
    if not os.path.isdir(_subdir):
      continue
    for img in os.listdir(_subdir):
      if img.endswith('.JPG'):
        fimage = edict()
        fimage.id = os.path.join(_subdir, img)
        fimage.classname = str(label)
        fimage.image_path = os.path.join(_subdir, img)
        json_file = fimage.image_path+".json"
        data = None
        fimage.bbox = None
        fimage.landmark = None
        if os.path.exists(json_file):
          with open(json_file, 'r') as f:
            data = f.read()
            data = json.loads(data)
          assert data is not None
          if 'bounding_box' in data:
            fimage.bbox = np.zeros( (4,), dtype=np.float32 )
            bb = data['bounding_box']
            fimage.bbox[0] = bb['x']
            fimage.bbox[1] = bb['y']
            fimage.bbox[2] = bb['x']+bb['width']
            fimage.bbox[3] = bb['y']+bb['height']
            #print('bb')
          if 'landmarks' in data:
            landmarks = data['landmarks']
            if '1' in landmarks and '0' in landmarks and '2' in landmarks:
              fimage.landmark = np.zeros( (3,2), dtype=np.float32 )
              fimage.landmark[0][0] = landmarks['1']['x']
              fimage.landmark[0][1] = landmarks['1']['y']
              fimage.landmark[1][0] = landmarks['0']['x']
              fimage.landmark[1][1] = landmarks['0']['y']
              fimage.landmark[2][0] = landmarks['2']['x']
              fimage.landmark[2][1] = landmarks['2']['y']
            #print('lm')

        #fimage.landmark = None
        ret.append(fimage)
    label+=1
  return ret

def get_dataset_ytf(input_dir):
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
    for _subdir2 in os.listdir(_subdir):
      _subdir2 = os.path.join(_subdir, _subdir2)
      if not os.path.isdir(_subdir2):
        continue
      _ret = []
      for img in os.listdir(_subdir2):
        fimage = edict()
        fimage.id = os.path.join(_subdir2, img)
        fimage.classname = str(label)
        fimage.image_path = os.path.join(_subdir2, img)
        fimage.bbox = None
        fimage.landmark = None
        _ret.append(fimage)
      ret += _ret
    label+=1
  return ret

def get_dataset_clfw(input_dir):
  ret = []
  label = 0
  for img in os.listdir(input_dir):
    fimage = edict()
    fimage.id = img
    fimage.classname = str(0)
    fimage.image_path = os.path.join(input_dir, img)
    fimage.bbox = None
    fimage.landmark = None
    ret.append(fimage)
  return ret

def get_dataset_common(input_dir, min_images = 1):
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
    _ret = []
    for img in os.listdir(_subdir):
      fimage = edict()
      fimage.id = os.path.join(person_name, img)
      fimage.classname = str(label)
      fimage.image_path = os.path.join(_subdir, img)
      fimage.bbox = None
      fimage.landmark = None
      _ret.append(fimage)
    if len(_ret)>=min_images:
      ret += _ret
      label+=1
  return ret

def get_dataset(name, input_dir):
  if name=='webface' or name=='lfw' or name=='vgg':
    return get_dataset_common(input_dir)
  if name=='celeb':
    return get_dataset_celeb(input_dir)
  if name=='facescrub':
    return get_dataset_facescrub(input_dir)
  if name=='megaface':
    return get_dataset_megaface(input_dir)
  if name=='fgnet':
    return get_dataset_fgnet(input_dir)
  if name=='ytf':
    return get_dataset_ytf(input_dir)
  if name=='clfw':
    return get_dataset_clfw(input_dir)
  return None


