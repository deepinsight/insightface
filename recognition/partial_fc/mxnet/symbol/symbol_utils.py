import sys
import os
import mxnet as mx

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from default import config


def Conv(**kwargs):
    # name = kwargs.get('name')
    # _weight = mx.symbol.Variable(name+'_weight')
    # _bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    # body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    # ignore param act_type, set it in this function
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body


bn_mom = config.bn_mom


def Linear(data,
           num_filter=1,
           kernel=(1, 1),
           stride=(1, 1),
           pad=(0, 0),
           num_group=1,
           name=None,
           suffix=''):
    conv = mx.sym.Convolution(data=data,
                              num_filter=num_filter,
                              kernel=kernel,
                              num_group=num_group,
                              stride=stride,
                              pad=pad,
                              no_bias=True,
                              name='%s%s_conv2d' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv,
                          name='%s%s_batchnorm' % (name, suffix),
                          fix_gamma=False,
                          momentum=bn_mom)
    return bn


def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
    body = last_conv
    if fc_type == 'Z':
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn1')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = body
    elif fc_type == 'E':
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn1')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body,
                                    num_hidden=num_classes,
                                    name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1,
                               fix_gamma=True,
                               eps=2e-5,
                               momentum=bn_mom,
                               name='fc1')
    elif fc_type == 'FC':
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn1')
        fc1 = mx.sym.FullyConnected(data=body,
                                    num_hidden=num_classes,
                                    name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1,
                               fix_gamma=True,
                               eps=2e-5,
                               momentum=bn_mom,
                               name='fc1')
    elif fc_type == 'SFC':
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn1')
        body = Conv(data=body,
                    num_filter=input_channel,
                    kernel=(3, 3),
                    stride=(2, 2),
                    pad=(1, 1),
                    no_bias=True,
                    name="convf",
                    num_group=input_channel)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bnf')
        body = Act(data=body, act_type=config.net_act, name='reluf')
        body = Conv(data=body,
                    num_filter=input_channel,
                    kernel=(1, 1),
                    pad=(0, 0),
                    stride=(1, 1),
                    name="convf2")
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bnf2')
        body = Act(data=body, act_type=config.net_act, name='reluf2')
        fc1 = mx.sym.FullyConnected(data=body,
                                    num_hidden=num_classes,
                                    name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1,
                               fix_gamma=True,
                               eps=2e-5,
                               momentum=bn_mom,
                               name='fc1')
    elif fc_type == 'GAP':
        bn1 = mx.sym.BatchNorm(data=body,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name='bn1')
        relu1 = Act(data=bn1, act_type=config.net_act, name='relu1')
        # Although kernel is not used here when global_pool=True, we should put one
        pool1 = mx.sym.Pooling(data=relu1,
                               global_pool=True,
                               kernel=(7, 7),
                               pool_type='avg',
                               name='pool1')
        flat = mx.sym.Flatten(data=pool1)
        fc1 = mx.sym.FullyConnected(data=flat,
                                    num_hidden=num_classes,
                                    name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1,
                               fix_gamma=True,
                               eps=2e-5,
                               momentum=bn_mom,
                               name='fc1')
    elif fc_type == 'GNAP':  # mobilefacenet++
        filters_in = 512  # param in mobilefacenet
        if num_classes > filters_in:
            body = mx.sym.Convolution(data=last_conv,
                                      num_filter=num_classes,
                                      kernel=(1, 1),
                                      stride=(1, 1),
                                      pad=(0, 0),
                                      no_bias=True,
                                      name='convx')
            body = mx.sym.BatchNorm(data=body,
                                    fix_gamma=False,
                                    eps=2e-5,
                                    momentum=0.9,
                                    name='convx_bn')
            body = Act(data=body, act_type=config.net_act, name='convx_relu')
            filters_in = num_classes
        else:
            body = last_conv
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=True,
                                eps=2e-5,
                                momentum=0.9,
                                name='bn6f')

        spatial_norm = body * body
        spatial_norm = mx.sym.sum(data=spatial_norm, axis=1, keepdims=True)
        spatial_sqrt = mx.sym.sqrt(spatial_norm)
        # spatial_mean=mx.sym.mean(spatial_sqrt, axis=(1,2,3), keepdims=True)
        spatial_mean = mx.sym.mean(spatial_sqrt)
        spatial_div_inverse = mx.sym.broadcast_div(spatial_mean, spatial_sqrt)

        spatial_attention_inverse = mx.symbol.tile(spatial_div_inverse,
                                                   reps=(1, filters_in, 1, 1))
        body = body * spatial_attention_inverse
        # body = mx.sym.broadcast_mul(body, spatial_div_inverse)

        fc1 = mx.sym.Pooling(body,
                             kernel=(7, 7),
                             global_pool=True,
                             pool_type='avg')
        if num_classes < filters_in:
            fc1 = mx.sym.BatchNorm(data=fc1,
                                   fix_gamma=True,
                                   eps=2e-5,
                                   momentum=0.9,
                                   name='bn6w')
            fc1 = mx.sym.FullyConnected(data=fc1,
                                        num_hidden=num_classes,
                                        name='pre_fc1')
        else:
            fc1 = mx.sym.Flatten(data=fc1)
        fc1 = mx.sym.BatchNorm(data=fc1,
                               fix_gamma=True,
                               eps=2e-5,
                               momentum=0.9,
                               name='fc1')
    elif fc_type == "GDC":  # mobilefacenet_v1
        conv_6_dw = Linear(last_conv,
                           num_filter=input_channel,
                           num_group=input_channel,
                           kernel=(7, 7),
                           pad=(0, 0),
                           stride=(1, 1),
                           name="conv_6dw7_7")
        conv_6_f = mx.sym.FullyConnected(data=conv_6_dw,
                                         num_hidden=num_classes,
                                         name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=conv_6_f,
                               fix_gamma=True,
                               eps=2e-5,
                               momentum=bn_mom,
                               name='fc1')
    elif fc_type == 'F':
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn1')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body,
                                    num_hidden=num_classes,
                                    name='fc1')
    elif fc_type == 'G':
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn1')
        fc1 = mx.sym.FullyConnected(data=body,
                                    num_hidden=num_classes,
                                    name='fc1')
    elif fc_type == 'H':
        fc1 = mx.sym.FullyConnected(data=body,
                                    num_hidden=num_classes,
                                    name='fc1')
    elif fc_type == 'I':
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn1')
        fc1 = mx.sym.FullyConnected(data=body,
                                    num_hidden=num_classes,
                                    name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1,
                               fix_gamma=True,
                               eps=2e-5,
                               momentum=bn_mom,
                               name='fc1')
    elif fc_type == 'J':
        fc1 = mx.sym.FullyConnected(data=body,
                                    num_hidden=num_classes,
                                    name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1,
                               fix_gamma=True,
                               eps=2e-5,
                               momentum=bn_mom,
                               name='fc1')
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
    # print('in unit3')
    bn1 = mx.sym.BatchNorm(data=data,
                           fix_gamma=False,
                           eps=2e-5,
                           momentum=bn_mom,
                           name=name + '_bn1')
    conv1 = Conv(data=bn1,
                 num_filter=num_filter,
                 kernel=(3, 3),
                 stride=(1, 1),
                 pad=(1, 1),
                 no_bias=True,
                 workspace=workspace,
                 name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1,
                           fix_gamma=False,
                           eps=2e-5,
                           momentum=bn_mom,
                           name=name + '_bn2')
    act1 = Act(data=bn2, act_type=config.net_act, name=name + '_relu1')
    conv2 = Conv(data=act1,
                 num_filter=num_filter,
                 kernel=(3, 3),
                 stride=stride,
                 pad=(1, 1),
                 no_bias=True,
                 workspace=workspace,
                 name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2,
                           fix_gamma=False,
                           eps=2e-5,
                           momentum=bn_mom,
                           name=name + '_bn3')

    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv(data=data,
                       num_filter=num_filter,
                       kernel=(1, 1),
                       stride=stride,
                       no_bias=True,
                       workspace=workspace,
                       name=name + '_conv1sc')
        shortcut = mx.sym.BatchNorm(data=conv1sc,
                                    fix_gamma=False,
                                    momentum=bn_mom,
                                    eps=2e-5,
                                    name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return bn3 + shortcut


def residual_unit_v1l(data, num_filter, stride, dim_match, name, bottle_neck):
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
    workspace = config.workspace
    bn_mom = config.bn_mom
    memonger = False
    use_se = config.net_se
    act_type = config.net_act
    # print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data,
                     num_filter=int(num_filter * 0.25),
                     kernel=(1, 1),
                     stride=(1, 1),
                     pad=(0, 0),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1,
                     num_filter=int(num_filter * 0.25),
                     kernel=(3, 3),
                     stride=(1, 1),
                     pad=(1, 1),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2,
                     num_filter=num_filter,
                     kernel=(1, 1),
                     stride=stride,
                     pad=(0, 0),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn3')

        if use_se:
            # se begin
            body = mx.sym.Pooling(data=bn3,
                                  global_pool=True,
                                  kernel=(7, 7),
                                  pool_type='avg',
                                  name=name + '_se_pool1')
            body = Conv(data=body,
                        num_filter=num_filter // 16,
                        kernel=(1, 1),
                        stride=(1, 1),
                        pad=(0, 0),
                        name=name + "_se_conv1",
                        workspace=workspace)
            body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
            body = Conv(data=body,
                        num_filter=num_filter,
                        kernel=(1, 1),
                        stride=(1, 1),
                        pad=(0, 0),
                        name=name + "_se_conv2",
                        workspace=workspace)
            body = mx.symbol.Activation(data=body,
                                        act_type='sigmoid',
                                        name=name + "_se_sigmoid")
            bn3 = mx.symbol.broadcast_mul(bn3, body)
            # se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=stride,
                           no_bias=True,
                           workspace=workspace,
                           name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc,
                                        fix_gamma=False,
                                        eps=2e-5,
                                        momentum=bn_mom,
                                        name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn3 + shortcut,
                   act_type=act_type,
                   name=name + '_relu3')
    else:
        conv1 = Conv(data=data,
                     num_filter=num_filter,
                     kernel=(3, 3),
                     stride=(1, 1),
                     pad=(1, 1),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1,
                               fix_gamma=False,
                               momentum=bn_mom,
                               eps=2e-5,
                               name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1,
                     num_filter=num_filter,
                     kernel=(3, 3),
                     stride=stride,
                     pad=(1, 1),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2,
                               fix_gamma=False,
                               momentum=bn_mom,
                               eps=2e-5,
                               name=name + '_bn2')
        if use_se:
            # se begin
            body = mx.sym.Pooling(data=bn2,
                                  global_pool=True,
                                  kernel=(7, 7),
                                  pool_type='avg',
                                  name=name + '_se_pool1')
            body = Conv(data=body,
                        num_filter=num_filter // 16,
                        kernel=(1, 1),
                        stride=(1, 1),
                        pad=(0, 0),
                        name=name + "_se_conv1",
                        workspace=workspace)
            body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
            body = Conv(data=body,
                        num_filter=num_filter,
                        kernel=(1, 1),
                        stride=(1, 1),
                        pad=(0, 0),
                        name=name + "_se_conv2",
                        workspace=workspace)
            body = mx.symbol.Activation(data=body,
                                        act_type='sigmoid',
                                        name=name + "_se_sigmoid")
            bn2 = mx.symbol.broadcast_mul(bn2, body)
            # se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=stride,
                           no_bias=True,
                           workspace=workspace,
                           name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc,
                                        fix_gamma=False,
                                        momentum=bn_mom,
                                        eps=2e-5,
                                        name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn2 + shortcut,
                   act_type=act_type,
                   name=name + '_relu3')


def get_head(data, version_input, num_filter):
    bn_mom = config.bn_mom
    workspace = config.workspace
    kwargs = {'bn_mom': bn_mom, 'workspace': workspace}
    data = data - 127.5
    data = data * 0.0078125
    # data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if version_input == 0:
        body = Conv(data=data,
                    num_filter=num_filter,
                    kernel=(7, 7),
                    stride=(2, 2),
                    pad=(3, 3),
                    no_bias=True,
                    name="conv0",
                    workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn0')
        body = Act(data=body, act_type=config.net_act, name='relu0')
        body = mx.sym.Pooling(data=body,
                              kernel=(3, 3),
                              stride=(2, 2),
                              pad=(1, 1),
                              pool_type='max')
    else:
        body = data
        _num_filter = min(num_filter, 64)
        body = Conv(data=body,
                    num_filter=_num_filter,
                    kernel=(3, 3),
                    stride=(1, 1),
                    pad=(1, 1),
                    no_bias=True,
                    name="conv0",
                    workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn0')
        body = Act(data=body, act_type=config.net_act, name='relu0')
        # body = residual_unit_v3(body, _num_filter, (2, 2), False, name='head', **kwargs)
        body = residual_unit_v1l(body,
                                 _num_filter, (2, 2),
                                 False,
                                 name='head',
                                 bottle_neck=False)
    return body
