import math, os
import argparse
import numpy as np
import oneflow as flow

from config import config, default, generate_val_config
import ofrecord_util
import validation_util
from symbols import fresnet100, fmobilefacenet

def get_val_args():
    val_parser = argparse.ArgumentParser(description="flags for validation")
    val_parser.add_argument("-network", default=default.network)
    args, rest = val_parser.parse_known_args()
    generate_val_config(args.network)
    for ds in config.val_targets:
        val_parser.add_argument(
            "--%s_dataset_dir" % ds,
            type=str,
            default=os.path.join(default.val_dataset_dir, ds),
            help="validation dataset dir",
        )
    val_parser.add_argument(
        "--val_data_part_num",
        type=str,
        default=default.val_data_part_num,
        help="validation dataset dir prefix",
    )
    val_parser.add_argument(
        "--lfw_total_images_num", type=int, default=12000, required=False
    )
    val_parser.add_argument(
        "--cfp_fp_total_images_num", type=int, default=14000, required=False
    )
    val_parser.add_argument(
        "--agedb_30_total_images_num", type=int, default=12000, required=False
    )

    # distribution config
    val_parser.add_argument(
        "--device_num_per_node",
        type=int,
        default=default.device_num_per_node,
        required=False,
    )
    val_parser.add_argument(
        "--num_nodes",
        type=int,
        default=default.num_nodes,
        help="node/machine number for training",
    )

    val_parser.add_argument(
        "--val_batch_size_per_device",
        default=default.val_batch_size_per_device,
        type=int,
        help="validation batch size per device",
    )
    val_parser.add_argument(
        "--nrof_folds", default=default.nrof_folds, type=int, help="nrof folds"
    )
    # model and log
    val_parser.add_argument(
        "--log_dir", type=str, default=default.log_dir, help="log info save"
    )
    val_parser.add_argument(
        "--model_load_dir", default=default.model_load_dir, help="path to load model."
    )
    return val_parser.parse_args()


def flip_data(images):
    images_flipped = np.flip(images, axis=2).astype(np.float32)
    return images_flipped


def get_val_config():
    config = flow.function_config()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    return config


class Validator(object):
    def __init__(self, args):
        self.args = args
        if default.do_validation_while_train:
            function_config = get_val_config()

            @flow.global_function(type="predict", function_config=function_config)
            def get_validation_datset_lfw_job():
                with flow.scope.placement("cpu", "0:0"):
                    issame, images = ofrecord_util.load_lfw_dataset(self.args)
                    return issame, images

            self.get_validation_datset_lfw_fn = get_validation_datset_lfw_job

            @flow.global_function(type="predict", function_config=function_config)
            def get_validation_dataset_cfp_fp_job():
                with flow.scope.placement("cpu", "0:0"):
                    issame, images = ofrecord_util.load_cfp_fp_dataset(self.args)
                    return issame, images

            self.get_validation_dataset_cfp_fp_fn = get_validation_dataset_cfp_fp_job

            @flow.global_function(type="predict", function_config=function_config)
            def get_validation_dataset_agedb_30_job():
                with flow.scope.placement("cpu", "0:0"):
                    issame, images = ofrecord_util.load_agedb_30_dataset(self.args)
                    return issame, images

            self.get_validation_dataset_agedb_30_fn = (
                get_validation_dataset_agedb_30_job
            )

            @flow.global_function(type="predict", function_config=function_config)
            def get_symbol_val_job(
                images: flow.typing.Numpy.Placeholder(
                    (self.args.val_batch_size_per_device, 112, 112, 3)
                )
            ):
                print("val batch data: ", images.shape)
                embedding = eval(config.net_name).get_symbol(images)
                return embedding

            self.get_symbol_val_fn = get_symbol_val_job

    def do_validation(self, dataset="lfw"):
        print("Validation on [{}]:".format(dataset))
        _issame_list = []
        _em_list = []
        _em_flipped_list = []
        batch_size = self.args.val_batch_size_per_device
        if dataset == "lfw":
            total_images_num = self.args.lfw_total_images_num
            val_job = self.get_validation_datset_lfw_fn
        if dataset == "cfp_fp":
            total_images_num = self.args.cfp_fp_total_images_num
            val_job = self.get_validation_dataset_cfp_fp_fn
        if dataset == "agedb_30":
            total_images_num = self.args.agedb_30_total_images_num
            val_job = self.get_validation_dataset_agedb_30_fn

        val_iter_num = math.ceil(total_images_num / batch_size)
        for i in range(val_iter_num):
            _issame, images = val_job().get()
            images_flipped = flip_data(images.numpy())
            _em = self.get_symbol_val_fn(images.numpy()).get()
            _em_flipped = self.get_symbol_val_fn(images_flipped).get()
            _issame_list.append(_issame.numpy())
            _em_list.append(_em.numpy())
            _em_flipped_list.append(_em_flipped.numpy())

        issame = np.array(_issame_list).flatten().reshape(-1, 1)[:total_images_num, :]
        issame_list = [bool(x) for x in issame[0::2]]
        embedding_length = _em_list[0].shape[-1]
        embeddings = (np.array(_em_list).flatten().reshape(-1, embedding_length))[
            :total_images_num, :
        ]
        embeddings_flipped = (
            np.array(_em_flipped_list).flatten().reshape(-1, embedding_length)
        )[:total_images_num, :]
        embeddings_list = [embeddings, embeddings_flipped]

        return issame_list, embeddings_list

    def load_checkpoint(self):
        flow.load_variables(flow.checkpoint.get(self.args.model_load_dir))


def main():
    args = get_val_args()
    flow.env.log_dir(args.log_dir)
    flow.config.gpu_device_num(args.device_num_per_node)

    # validation
    validator = Validator(args)
    validator.load_checkpoint()
    for ds in config.val_targets:
        issame_list, embeddings_list = validator.do_validation(dataset=ds)
        validation_util.cal_validation_metrics(
            embeddings_list, issame_list, nrof_folds=args.nrof_folds,
        )


if __name__ == "__main__":
    main()
