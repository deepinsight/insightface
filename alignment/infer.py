import argparse
import cv2
import numpy as np
import sys
import mxnet as mx

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='128,128', help='')
parser.add_argument('--model', default='./models/test,15', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
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
#model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
model.set_params(arg_params, aux_params)
#img_path = '/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54745.png'
img_path = './test.png'

img = cv2.imread(img_path)

rimg = cv2.resize(img, (image_size[1], image_size[0]))
img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
img = np.transpose(img, (2,0,1)) #3*112*112, RGB
input_blob = np.expand_dims(img, axis=0) #1*3*112*112
data = mx.nd.array(input_blob)
db = mx.io.DataBatch(data=(data,))
model.forward(db, is_train=False)
output = model.get_outputs()[0].asnumpy() 
#print(output[0,80])
#sys.exit(0)
filename = "./vis/draw_%s" % img_path.split('/')[-1]
for i in xrange(output.shape[1]):
  a = output[0,i,:,:]
  a = cv2.resize(a, (image_size[1], image_size[0]))
  ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
  cv2.circle(rimg, (ind[1], ind[0]), 1, (0, 0, 255), 2)
  print(i, ind)
cv2.imwrite(filename, rimg)

