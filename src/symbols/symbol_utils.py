import mxnet as mx

def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    return body


def get_fc1(last_conv, num_classes, fc_type):
  bn_mom = 0.9
  body = last_conv
  if fc_type=='E':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
  elif fc_type=='F':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
  elif fc_type=='G':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
  elif fc_type=='H':
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
  elif fc_type=='I':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
  elif fc_type=='J':
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
  else:
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = Act(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    if len(fc_type)>1:
      if fc_type[1]=='X':
        print('dropout mode')
        flat = mx.symbol.Dropout(data=flat, p=0.2)
      fc_type = fc_type[0]
    if fc_type=='A':
      fc1 = flat
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

def residual_unit_v3(data, num_filter, stride, dim_match, name, **kwargs):
    
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    #print('in unit3')
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act1 = Act(data=bn2, act_type='relu', name=name + '_relu1')
    conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_conv1sc')
        shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return bn3 + shortcut


def get_head(data, version_input, num_filter):
    bn_mom = 0.9
    workspace = 256
    kwargs = {'bn_mom': bn_mom, 'workspace' : workspace}
    data = data-127.5
    data = data*0.0078125
    #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if version_input==0:
      body = Conv(data=data, num_filter=num_filter, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      body = Act(data=body, act_type='relu', name='relu0')
      body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
      body = data
      _num_filter = min(num_filter, 64)
      body = Conv(data=body, num_filter=_num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      body = Act(data=body, act_type='relu', name='relu0')
      body = residual_unit_v3(body, _num_filter, (2, 2), False, name='head', **kwargs)
    return body


