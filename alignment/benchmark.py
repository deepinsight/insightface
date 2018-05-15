import argparse
import cv2
import numpy as np
import sys
import mxnet as mx
import datetime

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='128,128', help='')
parser.add_argument('--model', default='./models/test,15', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--batch-size', default=10, type=int, help='batch size')
parser.add_argument('--iterations', default=10, type=int, help='iterations')
args = parser.parse_args()

_vec = args.image_size.split(',')
assert len(_vec)==2
image_size = (int(_vec[0]), int(_vec[1]))
_vec = args.model.split(',')
assert len(_vec)==2
prefix = _vec[0]
epoch = int(_vec[1])
print('loading',prefix, epoch)
if args.gpu>=0:
  ctx = mx.gpu(args.gpu)
else:
  ctx = mx.cpu()
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers['heatmap_output']
model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
#model = mx.mod.Module(symbol=sym, context=ctx)
model.bind(for_training=False, data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
#model.bind(for_training=False, data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,84,64,64))])
model.set_params(arg_params, aux_params)
img_path = './test.png'

img = cv2.imread(img_path)

rimg = cv2.resize(img, (image_size[1], image_size[0]))
img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
img = np.transpose(img, (2,0,1)) #3*112*112, RGB
input_blob = np.zeros( (args.batch_size, 3, image_size[1], image_size[0]),dtype=np.uint8 )
for i in xrange(args.batch_size):
  input_blob[i] = img
data = mx.nd.array(input_blob)
print(data.shape)
label = mx.nd.zeros( (args.batch_size, 84, 64, 64) )
#db = mx.io.DataBatch(data=(data,))
db = mx.io.DataBatch(data=(data,), label=(label,))
stat = []
warmup = 2
for i in xrange(args.iterations+warmup):
  #print(i)
  time_now = datetime.datetime.now()
  model.forward(db, is_train=False)
  output = model.get_outputs()[-1].asnumpy() 
  time_now2 = datetime.datetime.now()
  diff = time_now2 - time_now
  stat.append(diff.total_seconds())
stat = stat[warmup:]
print(np.mean(stat)/args.batch_size)


