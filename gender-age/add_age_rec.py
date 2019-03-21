import mxnet as mx
import numpy as np
import sys, os

source_dir = sys.argv[1]
input_dir = sys.argv[2]
idx_file = os.path.join(source_dir, 'traino.idx')
rec_file = os.path.join(source_dir, 'traino.rec')
writer = mx.recordio.MXIndexedRecordIO(os.path.join(source_dir,'train.idx'), os.path.join(source_dir,'train.rec'), 'w')  # pylint: disable=redefined-variable-type
imgrec = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')  # pylint: disable=redefined-variable-type
seq = list(imgrec.keys)
widx = 0
for img_idx in seq:
  s = imgrec.read_idx(img_idx)
  assert widx==img_idx
  writer.write_idx(widx, s)
  widx+=1


stat = {}

for _file in os.listdir(input_dir):
  if not _file.endswith('.rec'):
    continue
  rec_file = os.path.join(input_dir, _file)
  print(rec_file)
  idx_file = rec_file[:-4]+'.idx'
  imgrec = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')  # pylint: disable=redefined-variable-type
  seq = list(imgrec.keys)
  for img_idx in seq:
    if img_idx%100==0:
      print(img_idx, stat)
    s = imgrec.read_idx(img_idx)
    header, img = mx.recordio.unpack(s)
    try:
      image = mx.image.imdecode(img).asnumpy()
    except:
      continue
    age = int(header.label[0])
    if age>=20:
      continue
    age_group = age//10
    #if not age in stat:
      stat[age_group] = 0
    stat[age_group]+=1
    label = [9999, age]
    nheader = mx.recordio.IRHeader(0, label, widx, 0)
    bgr = image[:,:,::-1]
    s = mx.recordio.pack_img(nheader, bgr, quality=95, img_fmt='.jpg')
    writer.write_idx(widx, s)
    widx+=1

