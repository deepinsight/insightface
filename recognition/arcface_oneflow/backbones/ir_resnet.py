import oneflow as flow
from .common import _batch_norm, _conv2d_layer, _avg_pool, _prelu, get_fc1


def residual_unit_v3(
    in_data, num_filter, stride, dim_match, bn_is_training, data_format, name
):

    suffix = ""
    use_se = 0
    bn1 = _batch_norm(
        in_data,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s.bn1" % (name, suffix),
    )
    conv1 = _conv2d_layer(
        name="%s%s.conv1" % (name, suffix),
        input=bn1,
        filters=num_filter,
        kernel_size=3,
        strides=[1, 1],
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn2 = _batch_norm(
        conv1,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s.bn2" % (name, suffix),
    )
    prelu = _prelu(bn2, data_format=data_format,
                   name="%s%s_relu1" % (name, suffix))
    conv2 = _conv2d_layer(
        name="%s%s.conv2" % (name, suffix),
        input=prelu,
        filters=num_filter,
        kernel_size=3,
        strides=stride,
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn3 = _batch_norm(
        conv2,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s.bn3" % (name, suffix),
    )

    if use_se:
        # se begin
        input_blob = _avg_pool(
            bn3, pool_size=[7, 7], strides=[1, 1], padding="VALID"
        )
        input_blob = _conv2d_layer(
            name="%s%s_se_conv1" % (name, suffix),
            input=input_blob,
            filters=num_filter // 16,
            kernel_size=1,
            strides=[1, 1],
            padding="valid",
            data_format=data_format,
            use_bias=True,
            dilation_rate=1,
            activation=None,
        )
        input_blob = _prelu(input_blob, name="%s%s_se_relu1" % (name, suffix))
        input_blob = _conv2d_layer(
            name="%s%s_se_conv2" % (name, suffix),
            input=input_blob,
            filters=num_filter,
            kernel_size=1,
            strides=[1, 1],
            padding="valid",
            data_format=data_format,
            use_bias=True,
            dilation_rate=1,
            activation=None,
        )
        input_blob = flow.math.sigmoid(input=input_blob)
        bn3 = flow.math.multiply(x=input_blob, y=bn3)
        # se end

    if dim_match:
        input_blob = in_data
    else:
        input_blob = _conv2d_layer(
            name="%s%s.downsample.0" % (name, suffix),
            input=in_data,
            filters=num_filter,
            kernel_size=1,
            strides=stride,
            padding="valid",
            data_format=data_format,
            use_bias=False,
            dilation_rate=1,
            activation=None,
        )
        input_blob = _batch_norm(
            input_blob,
            epsilon=2e-5,
            is_training=bn_is_training,
            data_format=data_format,
            name="%s%s.downsample.1" % (name, suffix),
        )

    identity = flow.math.add(x=bn3, y=input_blob)
    return identity


def get_symbol(input_blob, units, cfg):
    filter_list = [64, 64, 128, 256, 512]
    num_stages = 4
    units = units

    num_classes = cfg.embedding_size

    fc_type = cfg.fc_type
    bn_is_training = True
    data_format = "NCHW"

    input_blob = _conv2d_layer(
        name="conv1",
        input=input_blob,
        filters=filter_list[0],
        kernel_size=3,
        strides=[1, 1],
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    input_blob = _batch_norm(
        input_blob, epsilon=2e-5, is_training=bn_is_training, data_format=data_format, name="bn1"
    )
    input_blob = _prelu(input_blob, data_format=data_format, name="relu0")

    for i in range(num_stages):
        input_blob = residual_unit_v3(
            input_blob,
            filter_list[i + 1],
            [2, 2],
            False,
            bn_is_training=bn_is_training,
            data_format=data_format,
            name="layer%d.%d" % (i + 1, 0),
        )
        for j in range(units[i] - 1):
            input_blob = residual_unit_v3(
                input_blob,
                filter_list[i + 1],
                [1, 1],
                True,
                bn_is_training=bn_is_training,
                data_format=data_format,
                name="layer%d.%d" % (i + 1, j + 1),
            )
    fc1 = get_fc1(input_blob, num_classes, fc_type)
    return fc1


def iresnet18(input_blob, cfg):
    return get_symbol([2, 2, 2, 2], cfg)


def iresnet34(input_blob, cfg):
    return get_symbol(input_blob, [3, 4, 6, 3], cfg)


def iresnet50(input_blob, cfg):
    return get_symbol(input_blob,  [3, 4, 14, 3], cfg)


def iresnet100(input_blob, cfg):
    return get_symbol(input_blob,  [3, 13, 30, 3], cfg)


def iresnet200(input_blob, cfg):
    return get_symbol(input_blob,  [6, 26, 60, 6], cfg)
