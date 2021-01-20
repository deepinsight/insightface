import oneflow as flow
from symbols.symbol_utils import _batch_norm, _conv2d_layer, _avg_pool, _prelu, get_fc1
from config import config


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
        name="%s%s_bn1" % (name, suffix),
    )
    conv1 = _conv2d_layer(
        name="%s%s_conv1" % (name, suffix),
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
        name="%s%s_bn2" % (name, suffix),
    )
    prelu = _prelu(bn2, data_format=data_format,
                   name="%s%s_relu1" % (name, suffix))
    conv2 = _conv2d_layer(
        name="%s%s_conv2" % (name, suffix),
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
        name="%s%s_bn3" % (name, suffix),
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
            name="%s%s_conv1sc" % (name, suffix),
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
            name="%s%s_sc" % (name, suffix),
        )

    identity = flow.math.add(x=bn3, y=input_blob)
    return identity


def get_symbol(input_blob):
    filter_list = [64, 64, 128, 256, 512]
    num_stages = 4
    units = [3, 13, 30, 3]
    num_classes = config.emb_size
    fc_type = config.fc_type
    bn_is_training = config.bn_is_training
    data_format = config.data_format
    if data_format.upper() == "NCHW":
        input_blob = flow.transpose(
            input_blob, name="transpose", perm=[0, 3, 1, 2]
        )
    input_blob = _conv2d_layer(
        name="conv0",
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
        input_blob, epsilon=2e-5, is_training=bn_is_training, data_format=data_format, name="bn0"
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
            name="stage%d_unit%d" % (i + 1, 1),
        )
        for j in range(units[i] - 1):
            input_blob = residual_unit_v3(
                input_blob,
                filter_list[i + 1],
                [1, 1],
                True,
                bn_is_training=bn_is_training,
                data_format=data_format,
                name="stage%d_unit%d" % (i + 1, j + 2),
            )
    fc1 = get_fc1(input_blob, num_classes, fc_type)
    return fc1
