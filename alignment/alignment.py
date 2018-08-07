import argparse
import cv2
import numpy as np
import sys
import mxnet as mx
import datetime

class Alignment:
  def __init__(self, prefix, epoch, ctx_id=0):
    print('loading',prefix, epoch)
    ctx = mx.gpu(ctx_id)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['heatmap_output']
    image_size = (128, 128)
    self.image_size = image_size
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model = mx.mod.Module(symbol=sym, context=ctx)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
  
  def get(self, img):
    rimg = cv2.resize(img, (self.image_size[1], self.image_size[0]))
    img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2,0,1)) #3*112*112, RGB
    input_blob = np.zeros( (1, 3, self.image_size[1], self.image_size[0]),dtype=np.uint8 )
    input_blob[0] = img
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    alabel = self.model.get_outputs()[-1].asnumpy()[0]
    ret = np.zeros( (alabel.shape[0], 2), dtype=np.float32)
    for i in xrange(alabel.shape[0]):
      a = cv2.resize(alabel[i], (self.image_size[1], self.image_size[0]))
      ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
      #ret[i] = (ind[0], ind[1]) #h, w
      ret[i] = (ind[1], ind[0]) #w, h
    return ret



