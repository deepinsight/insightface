import oneflow as flow
from config import config


def _get_initializer():
    return flow.random_normal_initializer(mean=0.0, stddev=0.1)


def _get_regularizer(name):
    return None


def _dropout(input_blob, dropout_prob):
    return flow.nn.dropout(input_blob, rate=dropout_prob)


def _prelu(inputs, data_format="NCHW", name=None):
    return flow.layers.prelu(
        inputs,
        alpha_initializer=flow.constant_initializer(0.25),
        alpha_regularizer=_get_regularizer("alpha"),
        shared_axes=[2, 3] if data_format == "NCHW" else [1, 2],
        name=name,
    )


def _avg_pool(inputs, pool_size, strides, padding, data_format="NCHW", name=None):
    return flow.nn.avg_pool2d(
        input=inputs, ksize=pool_size, strides=strides, padding=padding, data_format=data_format, name=name
    )


def _batch_norm(
    inputs,
    epsilon,
    center=True,
    scale=True,
    trainable=True,
    is_training=True,
    data_format="NCHW",
    name=None,
):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=3 if data_format == "NHWC" and inputs.shape == 4 else 1,
        momentum=0.9,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=flow.zeros_initializer(),
        gamma_initializer=flow.ones_initializer(),
        beta_regularizer=_get_regularizer("beta"),
        gamma_regularizer=_get_regularizer("gamma"),
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=trainable,
        training=is_training,
        name=name,
    )


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    group_num=1,
    data_format="NCHW",
    dilation_rate=1,
    activation=None,
    use_bias=False,
    weight_initializer=_get_initializer(),
    bias_initializer=flow.zeros_initializer(),
    weight_regularizer=_get_regularizer("weight"),
    bias_regularizer=_get_regularizer("bias"),
):
    return flow.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, groups=group_num, activation=activation, use_bias=use_bias, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer, name=name)


def Linear(
    input_blob,
    num_filter=1,
    kernel=None,
    stride=None,
    pad="valid",
    num_group=1,
    bn_is_training=True,
    data_format="NCHW",
    name=None,
    suffix="",
):
    conv = _conv2d_layer(
        name="%s%s_conv2d" % (name, suffix),
        input=input_blob,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        data_format=data_format,
        group_num=num_group,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn = _batch_norm(
        conv,
        epsilon=0.001,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s_batchnorm" % (name, suffix),
    )
    return bn


def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
    body = last_conv
    if fc_type == "Z":
        body = _batch_norm(
            body,
            epsilon=2e-5,
            scale=False,
            center=True,
            is_training=config.bn_is_training,
            data_format=config.data_format,
            name="bn1"
        )
        body = _dropout(body, 0.4)
        fc1 = body
    elif fc_type == "E":
        body = _batch_norm(
            body,
            epsilon=2e-5,
            is_training=config.bn_is_training,
            data_format=config.data_format,
            name="bn1"
        )
        body = _dropout(body, dropout_prob=0.4)
        body = flow.reshape(body, (body.shape[0], -1))
        fc1 = flow.layers.dense(
            inputs=body,
            units=num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=_get_initializer(),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=_get_regularizer("weight"),
            bias_regularizer=_get_regularizer("bias"),
            trainable=True,
            name="pre_fc1",
        )
        fc1 = _batch_norm(
            fc1,
            epsilon=2e-5,
            scale=False,
            center=True,
            is_training=config.bn_is_training,
            data_format=config.data_format,
            name="fc1",
        )
    elif fc_type == "FC":
        body = _batch_norm(
            body,
            epsilon=2e-5,
            is_training=config.bn_is_training,
            data_format=config.data_format,
            name="bn1"
        )
        body = flow.reshape(body, (body.shape[0], -1))
        fc1 = flow.layers.dense(
            inputs=body,
            units=num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=_get_initializer(),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=_get_regularizer("weight"),
            bias_regularizer=_get_regularizer("bias"),
            trainable=True,
            name="pre_fc1"
        )
        fc1 = _batch_norm(
            fc1,
            epsilon=2e-5,
            scale=False,
            center=True,
            is_training=config.bn_is_training,
            data_format=config.data_format,
            name="fc1"
        )
    elif fc_type == "GDC":
        conv_6_dw = Linear(
            last_conv,
            num_filter=input_channel,  # 512
            num_group=input_channel,  # 512
            kernel=7,
            pad="valid",
            stride=[1, 1],
            bn_is_training=config.bn_is_training,
            data_format=config.data_format,
            name="conv_6dw7_7",
        )
        conv_6_dw = flow.reshape(conv_6_dw, (body.shape[0], -1))
        conv_6_f = flow.layers.dense(
            inputs=conv_6_dw,
            units=num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=_get_initializer(),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=_get_regularizer("weight"),
            bias_regularizer=_get_regularizer("bias"),
            trainable=True,
            name="pre_fc1",
        )
        fc1 = _batch_norm(
            conv_6_f,
            epsilon=2e-5,
            scale=False,
            center=True,
            is_training=config.bn_is_training,
            data_format=config.data_format,
            name="fc1",
        )
    return fc1
