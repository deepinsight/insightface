import os
import oneflow as flow


def train_dataset_reader(
    args, data_dir, batch_size, data_part_num, part_name_suffix_length=1
):
    if os.path.exists(data_dir):
        print("Loading train data from {}".format(data_dir))
    else:
        raise Exception("Invalid train dataset dir", data_dir)
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(112, 112, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec(
            image_preprocessors=[
                flow.data.ImagePreprocessor("bgr2rgb"),
                flow.data.ImagePreprocessor("mirror"),
            ]
        ),
        preprocessors=[
            flow.data.NormByChannelPreprocessor(
                mean_values=(127.5, 127.5, 127.5), std_values=(127.5, 127.5, 127.5), data_format="NCHW"
            ),
        ],
    )

    label_blob_conf = flow.data.BlobConf(
        "label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir,
        (label_blob_conf, image_blob_conf),
        batch_size=batch_size,
        data_part_num=data_part_num,
        part_name_prefix=args.part_name_prefix,
        part_name_suffix_length=args.part_name_suffix_length,
        shuffle=args.shuffle,
        buffer_size=16384,
    )


def load_synthetic(config):
    batch_size = config.train_batch_size
    image_size = 112
    label = flow.data.decode_random(
        shape=(),
        dtype=flow.int32,
        batch_size=batch_size,
        initializer=flow.zeros_initializer(flow.int32),
    )

    image = flow.data.decode_random(
        shape=(image_size, image_size, 3), dtype=flow.float, batch_size=batch_size,
    )
    return label, image


def load_train_dataset(args):
    data_dir = args.ofrecord_path
    batch_size = args.total_batch_size
    data_part_num = args.train_data_part_num
    part_name_suffix_length = args.part_name_suffix_length
    print("train batch size in load train dataset: ", batch_size)
    labels, images = train_dataset_reader(
        args, data_dir, batch_size, data_part_num, part_name_suffix_length
    )
    return labels, images
