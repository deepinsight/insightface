import mxnet as mx
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="flags for convert model")
parser.add_argument(
    "-mx_load_prefix", "--mxnet_load_prefix", type=str, required=False
)
parser.add_argument(
    "-mx_load_epoch", "--mxnet_load_epoch", type=int, required=False
)
parser.add_argument("-of", "--of_model_dir", type=str, required=False)

args = parser.parse_args()
assert not os.path.exists(args.of_model_dir)
os.mkdir(args.of_model_dir)

prefix = args.mxnet_load_prefix
epoch = args.mxnet_load_epoch
of_model_path = args.of_model_dir
assert not os.path.exists(of_model_path)
os.mkdir(of_model_path)


def _SaveWeightBlob2File(blob, folder, var):
    filename = os.path.join(folder, var)
    f = open(filename, "w")
    f.write(blob.tobytes())
    f.close()

    os.mkdir(folder + "-momentum")
    filename_momentum = os.path.join(folder + "-momentum", var)
    f2 = open(filename_momentum, "w")
    momentum = np.zeros(blob.shape, dtype=np.float32)
    f2.write(momentum.tobytes())
    f2.close()


_, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
for param_name in arg_params.keys():
    output_path = os.path.join(
        of_model_path, "_".join(param_name.split("_")[0:-1])
    )
    weight = arg_params[param_name].asnumpy()
    if param_name.split("_")[-2] == "conv2d":
        os.mkdir(output_path + "-" + param_name.split("_")[-1])
        if param_name.split("_")[-1] == "weight":
            _SaveWeightBlob2File(weight, output_path + "-weight", "out")
        elif param_name.split("_")[-1] == "bias":
            _SaveWeightBlob2File(weight, output_path + "-bias", "out")
        else:
            print(param_name, "error error")

    elif param_name.split("_")[-2] == "batchnorm":
        os.mkdir(output_path + "-" + param_name.split("_")[-1])
        if param_name.split("_")[-1] == "gamma":
            _SaveWeightBlob2File(weight, output_path + "-gamma", "out")
        elif param_name.split("_")[-1] == "beta":
            _SaveWeightBlob2File(weight, output_path + "-beta", "out")
        else:
            print(param_name, "error error")
    elif param_name.split("_")[-2] == "relu":
        if param_name.split("_")[-1] == "gamma":
            os.mkdir(output_path + "-alpha")
            _SaveWeightBlob2File(weight, output_path + "-alpha", "out")
        else:
            print(param_name, "error error")
    elif param_name.split("_")[-2] == "0":
        output_path = os.path.join(
            of_model_path, "_".join(param_name.split("_")[0:-2])
        )
        os.mkdir(output_path + "-" + param_name.split("_")[-1])
        if param_name.split("_")[-1] == "weight":
            _SaveWeightBlob2File(weight, output_path + "-weight", "out")
        elif param_name.split("_")[-1] == "bias":
            _SaveWeightBlob2File(weight, output_path + "-bias", "out")
        else:
            print("error error")
    elif param_name.split("_")[-2] == "fc1":
        os.mkdir(output_path + "-" + param_name.split("_")[-1])
        if param_name.split("_")[-1] == "weight":
            _SaveWeightBlob2File(weight, output_path + "-weight", "out")
        elif param_name.split("_")[-1] == "bias":
            _SaveWeightBlob2File(weight, output_path + "-bias", "out")
        elif param_name.split("_")[-1] == "beta":
            _SaveWeightBlob2File(weight, output_path + "-beta", "out")
        elif param_name.split("_")[-1] == "gamma":
            _SaveWeightBlob2File(weight, output_path + "-gamma", "out")
        else:
            print(param_name, "error error")
    else:
        print(param_name, "error error")

for param_name in aux_params.keys():
    output_path = os.path.join(
        of_model_path, "_".join(param_name.split("_")[0:-2])
    )
    weight = aux_params[param_name].asnumpy()
    if (
        param_name.split("_")[-3] == "batchnorm"
        or param_name.split("_")[-3] == "fc1"
    ):
        if param_name.split("_")[-1] == "mean":
            os.mkdir(output_path + "-moving_mean")
            _SaveWeightBlob2File(weight, output_path + "-moving_mean", "out")
        elif param_name.split("_")[-1] == "var":
            os.mkdir(output_path + "-moving_variance")
            _SaveWeightBlob2File(
                weight, output_path + "-moving_variance", "out"
            )
        else:
            print(param_name, "error error")
    else:
        print(param_name, "error error")

global_step = os.path.join(
    of_model_path, "System-Train-TrainStep-insightface_train_job"
)
os.mkdir(global_step)
f = open(os.path.join(global_step, "out"), "w")
step = np.zeros(2, dtype=np.float32)
f.write(step.tobytes())
f.close()
