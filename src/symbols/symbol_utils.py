import mxnet as mx

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    return body


def get_fc1(last_conv, num_classes, fc_type):
  bn_mom = 0.9
  body = last_conv
  if fc_type=='E':
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
  elif fc_type=='F':
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    fc1 = Act(data=fc1, act_type='relu', name='fc1_relu')
  else:
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = Act(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    if fc_type=='A':
      fc1 = flat
    else:
      if fc_type=='G' or fc_type=='H':
        fc1 = mx.symbol.Dropout(data=flat, p=0.2)
        fc1 = mx.sym.FullyConnected(data=fc1, num_hidden=num_classes, name='pre_fc1')
        if fc_type=='H':
          fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
      else:
        #B-D
        #B
        fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='pre_fc1')
        if fc_type=='C':
          fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
        elif fc_type=='D':
          fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
          fc1 = Act(data=fc1, act_type='relu', name='fc1_relu')
  return fc1

