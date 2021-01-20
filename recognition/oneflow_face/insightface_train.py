import os
import math
import argparse
import numpy as np
import oneflow as flow

from config import config, default, generate_config
import ofrecord_util
import validation_util
from callback_util import TrainMetric
from insightface_val import Validator, get_val_args

from symbols import fresnet100, fmobilefacenet


def str2list(x):
    x = [float(y) if type(eval(y)) == float else int(y) for y in x.split(',')]
    return x


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def get_train_args():
    train_parser = argparse.ArgumentParser(description="Flags for train")
    train_parser.add_argument(
        "--dataset", default=default.dataset, required=True, help="Dataset config"
    )
    train_parser.add_argument(
        "--network", default=default.network, required=True, help="Network config"
    )
    train_parser.add_argument(
        "--loss", default=default.loss, required=True, help="Loss config")
    args, rest = train_parser.parse_known_args()
    generate_config(args.network, args.dataset, args.loss)

    # distribution config
    train_parser.add_argument(
        "--device_num_per_node",
        type=int,
        default=default.device_num_per_node,
        help="The number of GPUs used per node",
    )
    train_parser.add_argument(
        "--num_nodes",
        type=int,
        default=default.num_nodes,
        help="Node/Machine number for training",
    )
    train_parser.add_argument(
        "--node_ips",
        type=str2list,
        default=default.node_ips,
        help='Nodes ip list for training, devided by ",", length >= num_nodes',
    )
    train_parser.add_argument(
        "--model_parallel",
        type=str2bool,
        nargs="?",
        default=default.model_parallel,
        help="Whether to use model parallel",
    )
    train_parser.add_argument(
        "--partial_fc",
        type=str2bool,
        nargs="?",
        default=default.partial_fc,
        help="Whether to use partial fc",
    )

    # train config
    train_parser.add_argument(
        "--train_batch_size",
        type=int,
        default=default.train_batch_size,
        help="Train batch size totally",
    )
    train_parser.add_argument(
        "--use_synthetic_data",
        type=str2bool,
        nargs="?",
        default=default.use_synthetic_data,
        help="Whether to use synthetic data",
    )
    train_parser.add_argument(
        "--do_validation_while_train",
        type=str2bool,
        nargs="?",
        default=default.do_validation_while_train,
        help="Whether do validation while training",
    )
    train_parser.add_argument(
        "--use_fp16", type=str2bool, nargs="?", default=default.use_fp16, help="Whether to use fp16"
    )
    train_parser.add_argument("--nccl_fusion_threshold_mb", type=int, default=default.nccl_fusion_threshold_mb,
                              help="NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow.")
    train_parser.add_argument("--nccl_fusion_max_ops", type=int, default=default.nccl_fusion_max_ops,
                              help="Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow.")

    # hyperparameters
    train_parser.add_argument(
        "--train_unit",
        type=str,
        default=default.train_unit,
        help="Choose train unit of iteration, batch or epoch",
    )
    train_parser.add_argument(
        "--train_iter",
        type=int,
        default=default.train_iter,
        help="Iteration for training",
    )
    train_parser.add_argument(
        "--lr", type=float, default=default.lr, help="Initial start learning rate"
    )
    train_parser.add_argument(
        "--lr_steps",
        type=str2list,
        default=default.lr_steps,
        help="Steps of lr changing",
    )
    train_parser.add_argument(
        "-wd", "--weight_decay", type=float, default=default.wd, help="Weight decay"
    )
    train_parser.add_argument(
        "-mom", "--momentum", type=float, default=default.mom, help="Momentum"
    )
    train_parser.add_argument("--scales", type=str2list,
                              default=default.scales, help="Learning rate step sacles")

    # model and log
    train_parser.add_argument(
        "--model_load_dir",
        type=str,
        default=default.model_load_dir,
        help="Path to load model",
    )
    train_parser.add_argument(
        "--models_root",
        type=str,
        default=default.models_root,
        help="Root directory to save model.",
    )
    train_parser.add_argument(
        "--log_dir", type=str, default=default.log_dir, help="Log info save directory"
    )

    train_parser.add_argument(
        "--loss_print_frequency",
        type=int,
        default=default.loss_print_frequency,
        help="Frequency of printing loss",
    )
    train_parser.add_argument(
        "--iter_num_in_snapshot",
        type=int,
        default=default.iter_num_in_snapshot,
        help="The number of train unit iter in the snapshot",
    )
    train_parser.add_argument(
        "--sample_ratio",
        type=float,
        default=default.sample_ratio,
        help="The ratio for sampling",
    )

    # validation config
    train_parser.add_argument(
        "--val_batch_size_per_device",
        type=int,
        default=default.val_batch_size_per_device,
        help="Validation batch size per device",
    )
    train_parser.add_argument(
        "--validation_interval",
        type=int,
        default=default.validation_interval,
        help="Validation interval while training, using train unit as interval unit",
    )
    train_parser.add_argument(
        "--val_data_part_num",
        type=str,
        default=default.val_data_part_num,
        help="Validation dataset dir prefix",
    )
    train_parser.add_argument(
        "--lfw_total_images_num", type=int, default=12000,
    )
    train_parser.add_argument(
        "--cfp_fp_total_images_num", type=int, default=14000,
    )
    train_parser.add_argument(
        "--agedb_30_total_images_num", type=int, default=12000,
    )
    for ds in config.val_targets:
        assert ds == 'lfw' or 'cfp_fp' or 'agedb_30', "Lfw, cfp_fp, agedb_30 datasets are supported now!"
        train_parser.add_argument(
            "--%s_dataset_dir" % ds,
            type=str,
            default=os.path.join(default.val_dataset_dir, ds),
            help="Validation dataset path",
        )
    train_parser.add_argument(
        "--nrof_folds", type=int, default=default.nrof_folds,
    )
    return train_parser.parse_args()


def get_train_config(args):
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)
    func_config.cudnn_conv_heuristic_search_algo(
        config.cudnn_conv_heuristic_search_algo
    )

    func_config.enable_fuse_model_update_ops(
        config.enable_fuse_model_update_ops)
    func_config.enable_fuse_add_to_output(config.enable_fuse_add_to_output)
    if args.use_fp16:
        print("Training with FP16 now.")
        func_config.enable_auto_mixed_precision(True)
    if args.partial_fc:
        func_config.enable_fuse_model_update_ops(False)
        func_config.indexed_slices_optimizer_conf(
            dict(include_op_names=dict(op_name=['fc7-weight'])))
    if args.use_fp16 and (args.num_nodes * args.device_num_per_node) > 1:
        flow.config.collective_boxing.nccl_fusion_all_reduce_use_buffer(False)
    if args.nccl_fusion_threshold_mb:
        flow.config.collective_boxing.nccl_fusion_threshold_mb(
            args.nccl_fusion_threshold_mb)
    if args.nccl_fusion_max_ops:
        flow.config.collective_boxing.nccl_fusion_max_ops(
            args.nccl_fusion_max_ops)
    size = args.device_num_per_node * args.num_nodes
    num_local = (config.num_classes + size - 1) // size
    num_sample = int(num_local * args.sample_ratio)
    args.total_num_sample = num_sample * size

    assert args.train_iter > 0, "Train iter must be greater than 0!"
    steps_per_epoch = math.ceil(config.total_img_num / args.train_batch_size)
    if args.train_unit == "epoch":
        print("Using epoch as training unit now. Each unit of iteration is epoch, including train_iter, iter_num_in_snapshot and validation interval")
        args.total_iter_num = steps_per_epoch * args.train_iter
        args.iter_num_in_snapshot = steps_per_epoch * args.iter_num_in_snapshot
        if args.validation_interval <= args.total_iter_num:
            args.validation_interval = steps_per_epoch * args.validation_interval
        else:
            print(
                "It doesn't do validation because validation_interval is greater than train_iter.")
    elif args.train_unit == "batch":
        print("Using batch as training unit now. Each unit of iteration is batch, including train_iter, iter_num_in_snapshot and validation interval")
        args.total_iter_num = args.train_iter
        args.iter_num_in_snapshot = args.iter_num_in_snapshot
        args.validation_interval = args.validation_interval
    else:
        raise ValueError("Invalid train unit!")
    return func_config


def make_train_func(args):
    @flow.global_function(type="train", function_config=get_train_config(args))
    def get_symbol_train_job():
        if args.use_synthetic_data:
            (labels, images) = ofrecord_util.load_synthetic(args)
        else:
            labels, images = ofrecord_util.load_train_dataset(args)
        image_size = images.shape[1:-1]
        assert len(
            image_size) == 2, "The length of image size must be equal to 2."
        assert image_size[0] == image_size[1], "image_size[0] should be equal to image_size[1]."
        print("train image_size: ", image_size)
        embedding = eval(config.net_name).get_symbol(images)

        def _get_initializer():
            return flow.random_normal_initializer(mean=0.0, stddev=0.01)

        trainable = True
        if config.loss_name == "softmax":
            if args.model_parallel:
                print("Training is using model parallelism now.")
                labels = labels.with_distribute(flow.distribute.broadcast())
                fc1_distribute = flow.distribute.broadcast()
                fc7_data_distribute = flow.distribute.split(1)
                fc7_model_distribute = flow.distribute.split(0)
            else:
                fc1_distribute = flow.distribute.split(0)
                fc7_data_distribute = flow.distribute.split(0)
                fc7_model_distribute = flow.distribute.broadcast()

            fc7 = flow.layers.dense(
                inputs=embedding.with_distribute(fc1_distribute),
                units=config.num_classes,
                activation=None,
                use_bias=False,
                kernel_initializer=_get_initializer(),
                bias_initializer=None,
                trainable=trainable,
                name="fc7",
                model_distribute=fc7_model_distribute,
            )
            fc7 = fc7.with_distribute(fc7_data_distribute)
        elif config.loss_name == "margin_softmax":
            if args.model_parallel:
                print("Training is using model parallelism now.")
                labels = labels.with_distribute(flow.distribute.broadcast())
                fc1_distribute = flow.distribute.broadcast()
                fc7_data_distribute = flow.distribute.split(1)
                fc7_model_distribute = flow.distribute.split(0)
            else:
                fc1_distribute = flow.distribute.split(0)
                fc7_data_distribute = flow.distribute.split(0)
                fc7_model_distribute = flow.distribute.broadcast()
            fc7_weight = flow.get_variable(
                name="fc7-weight",
                shape=(config.num_classes, embedding.shape[1]),
                dtype=embedding.dtype,
                initializer=_get_initializer(),
                regularizer=None,
                trainable=trainable,
                model_name="weight",
                distribute=fc7_model_distribute,
            )
            if args.partial_fc and args.model_parallel:
                print(
                    "Training is using model parallelism and optimized by partial_fc now."
                )
                (
                    mapped_label,
                    sampled_label,
                    sampled_weight,
                ) = flow.distributed_partial_fc_sample(
                    weight=fc7_weight, label=labels, num_sample=args.total_num_sample,
                )
                labels = mapped_label
                fc7_weight = sampled_weight
            fc7_weight = flow.math.l2_normalize(
                input=fc7_weight, axis=1, epsilon=1e-10)
            fc1 = flow.math.l2_normalize(
                input=embedding, axis=1, epsilon=1e-10)
            fc7 = flow.matmul(
                a=fc1.with_distribute(fc1_distribute), b=fc7_weight, transpose_b=True
            )
            fc7 = fc7.with_distribute(fc7_data_distribute)
            fc7 = (
                flow.combined_margin_loss(
                    fc7, labels, m1=config.loss_m1, m2=config.loss_m2, m3=config.loss_m3
                )
                * config.loss_s
            )
            fc7 = fc7.with_distribute(fc7_data_distribute)
        else:
            raise NotImplementedError

        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, fc7, name="softmax_loss"
        )

        lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(
            base_lr=args.lr,
            boundaries=args.lr_steps,
            scale=args.scales,
            warmup=None
        )
        flow.optimizer.SGDW(lr_scheduler,
                            momentum=args.momentum if args.momentum > 0 else None,
                            weight_decay=args.weight_decay
                            ).minimize(loss)

        return loss

    return get_symbol_train_job


def main(args):

    flow.config.gpu_device_num(args.device_num_per_node)
    print("gpu num: ", args.device_num_per_node)
    if not os.path.exists(args.models_root):
        os.makedirs(args.models_root)
    prefix = os.path.join(
        args.models_root, "%s-%s-%s" % (args.network,
                                        args.loss, args.dataset), "model"
    )
    prefix_dir = os.path.dirname(prefix)
    print("prefix: ", prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)

    if args.num_nodes > 1:
        assert args.num_nodes <= len(
            args.node_ips), "The number of nodes should not be greater than length of node_ips list."
        flow.env.ctrl_port(12138)
        nodes = []
        for ip in args.node_ips:
            addr_dict = {}
            addr_dict["addr"] = ip
            nodes.append(addr_dict)

        flow.env.machine(nodes)
    if config.data_format.upper() != "NCHW" and config.data_format.upper() != "NHWC":
        raise ValueError("Invalid data format")
    flow.env.log_dir(args.log_dir)
    train_func = make_train_func(args)
    validator = Validator(args)
    if os.path.exists(args.model_load_dir):
        print("Loading model from {}".format(args.model_load_dir))
        variables = flow.checkpoint.get(args.model_load_dir)
        flow.load_variables(variables)

    print("num_classes ", config.num_classes)
    print("Called with argument: ", args, config)
    train_metric = TrainMetric(
        desc="train", calculate_batches=args.loss_print_frequency, batch_size=args.train_batch_size
    )
    lr = args.lr

    for step in range(args.total_iter_num):
        # train
        train_func().async_get(train_metric.metric_cb(step))

        # validation
        if args.do_validation_while_train and (step + 1) % args.validation_interval == 0:
            for ds in config.val_targets:
                issame_list, embeddings_list = validator.do_validation(
                    dataset=ds)
                validation_util.cal_validation_metrics(
                    embeddings_list, issame_list, nrof_folds=args.nrof_folds,
                )
        if step in args.lr_steps:
            lr *= 0.1
            print("lr_steps: ", step)
            print("lr change to ", lr)

        # snapshot
        if (step + 1) % args.iter_num_in_snapshot == 0:
            path = os.path.join(
                prefix_dir, "snapshot_" + str(step // args.iter_num_in_snapshot))
            flow.checkpoint.save(path)


if __name__ == "__main__":
    args = get_train_args()
    main(args)
