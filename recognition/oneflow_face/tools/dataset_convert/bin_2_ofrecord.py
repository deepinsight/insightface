import os
import sys
import argparse
import pickle
import struct


import oneflow.core.record.record_pb2 as of_record


def parse_arguement(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/home/qiaojing/git_repo_out/insightface/datasets/faces_emore',
                        help='Root directory to mxnet dataset.')
    parser.add_argument('--output_filepath', type=str, default='./output',
                        help='Path to output OFRecord.')
    parser.add_argument('--dataset_name', type=str, default='lfw',
                        help='dataset_name.')
    return parser.parse_args(argv)


def load_bin_data(data_dir, dataset_name):
    path = os.path.join(data_dir, dataset_name+".bin")
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
        return bins, issame_list


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

    example = of_record.OFRecord(feature={
        'issame': _int32_feature(img_data['label']),
        'encoded': _bytes_feature(img_data['pixel_data']),
    })

    return example


def main(args):
    # Convert bin to ofrecord
    bins, issame_list = load_bin_data(
        data_dir=args.data_dir, dataset_name=args.dataset_name)
    output_dir = args.output_filepath
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'part-0')
    with open(output_file, 'wb') as f:
        for idx in range(len(bins)):
            if idx % 1000 == 0:
                print("Converting images: {} of {}".format(idx, len(bins)))
            img_data = {}
            img_data['label'] = int(issame_list[idx // 2])
            if args.dataset_name == "lfw":
                img_data['pixel_data'] = bins[idx]
            elif args.dataset_name == "cfp_fp" or args.dataset_name == "agedb_30":
                img_data['pixel_data'] = bins[idx].tobytes()
            else:
                raise NotImplementedError

            example = convert_to_ofrecord(img_data)
            l = example.ByteSize()
            f.write(struct.pack("q", l))
            f.write(example.SerializeToString())


if __name__ == '__main__':
    main(parse_arguement(sys.argv[1:]))
