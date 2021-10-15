import os
import sys
import struct
import argparse

from mxnet import recordio
import oneflow.core.record.record_pb2 as of_record


def parse_arguement(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="insightface/datasets/faces_emore",
        help="Root directory to mxnet dataset.",
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        default="./output",
        help="Path to output OFRecord.",
    )
    return parser.parse_args(argv)


def load_train_data(data_dir):

    path_imgrec = os.path.join(data_dir, "train.rec")
    path_imgidx = path_imgrec[0:-4] + ".idx"

    print(
        "Loading recordio {}\n\
  Corresponding record idx is {}".format(
            path_imgrec, path_imgidx
        )
    )

    imgrec = recordio.MXIndexedRecordIO(
        path_imgidx, path_imgrec, "r", key_type=int
    )
    # TODO: key_type ??

    # Read header0 to get some info.
    identity_key_start = 0
    identity_key_end = 0
    imgidx_list = []
    id2range = {}

    rec0 = imgrec.read_idx(0)
    header0, img_str = recordio.unpack(rec0)
    if header0.flag > 0:
        identity_key_start = int(header0.label[0])
        identity_key_end = int(header0.label[1])
        imgidx_list = range(1, identity_key_start)

        # Read identity id range
        for identity in range(identity_key_start, identity_key_end):
            rec = imgrec.read_idx(identity)
            header, s = recordio.unpack(rec)
            a, b = int(header.label[0]), int(header.label[1])
            id2range[identity] = (a, b)

    else:
        imgidx_list = imgrec.keys

    # print id2range to txt file
    # with open('id2range.txt', 'w') as f:
    #   for identity in range(identity_key_start, identity_key_end):
    #     l = str(identity) \
    #         + ' ' \
    #         + str(id2range[identity][0]) \
    #         + ' ' + str(id2range[identity][1]) + '\n'
    #     f.write(l)
    return imgrec, imgidx_list


def convert_to_ofrecord(img_data):
    """ Convert python dictionary formath data of one image to of.Example proto.
  Args:
    img_data: Python dict.
  Returns:
    example: The converted of.Exampl
  """

    def _int32_feature(value):
        """Wrapper for inserting int32 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return of_record.Feature(int32_list=of_record.Int32List(value=value))

    def _float_feature(value):
        """Wrapper for inserting float features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return of_record.Feature(float_list=of_record.FloatList(value=value))

    def _double_feature(value):
        """Wrapper for inserting float features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return of_record.Feature(double_list=of_record.DoubleList(value=value))

    def _bytes_feature(value):
        """Wrapper for inserting bytes features into Example proto."""
        # if isinstance(value, six.string_types):
        #  value = six.binary_type(value, encoding='utf-8')
        return of_record.Feature(bytes_list=of_record.BytesList(value=[value]))

    example = of_record.OFRecord(
        feature={
            "label": _int32_feature(img_data["label"]),
            "encoded": _bytes_feature(img_data["pixel_data"]),
        }
    )

    return example


def main(args):
    # Convert recordio to ofrecord
    imgrec, imgidx_list = load_train_data(data_dir=args.data_dir)

    output_dir = os.path.join(args.output_filepath, "train")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "part-0")
    with open(output_file, "wb") as f:
        for idx in imgidx_list:
            if idx % 10000 == 0:
                print(
                    "Converting images: {} of {}".format(
                        idx, len(imgidx_list)
                    )
                )

            img_data = {}
            rec = imgrec.read_idx(idx)
            header, s = recordio.unpack(rec)
            img_data["label"] = int(header.label[0])
            img_data["pixel_data"] = s

            example = convert_to_ofrecord(img_data)
            print("shape", len(img_data["pixel_data"]))
            size = example.ByteSize()
            f.write(struct.pack("q", size))
            f.write(example.SerializeToString())


if __name__ == "__main__":
    main(parse_arguement(sys.argv[1:]))
