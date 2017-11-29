
from easydict import EasyDict as edict
import os

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
          ret.append(fimage)
      label+=1
  return ret

def get_dataset_common(input_dir):
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
    for img in os.listdir(_subdir):
      fimage = edict()
      fimage.id = os.path.join(person_name, img)
      fimage.classname = str(label)
      fimage.image_path = os.path.join(_subdir, img)
      fimage.bbox = None
      fimage.landmark = None
      ret.append(fimage)
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
  return None


