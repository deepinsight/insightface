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


of_dump_path = args.of_model_dir
prefix = args.mxnet_load_prefix
# prefix = 'pretrained_LResNet100E/model-r100-ii/model'
epoch = args.mxnet_load_epoch  # 0
_, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)


def _SaveWeightBlob2File(blob, folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    filename = os.path.join(folder, "out")
    f = open(filename, "w")
    f.write(blob.tobytes())
    f.close()
    os.mkdir(folder + "-momentum")
    filename_momentum = os.path.join(folder + "-momentum", "out")
    f2 = open(filename_momentum, "w")
    momentum = np.zeros(blob.shape, dtype=np.float32)
    f2.write(momentum.tobytes())
    f2.close()


print("arg_params")
for param_name in arg_params.keys():
    output_path = of_dump_path + "_".join(param_name.split("_")[0:-1])

    weight = arg_params[param_name].asnumpy()
    if "conv" in param_name.split("_")[-2]:
        if param_name.split("_")[-1] == "weight":
            _SaveWeightBlob2File(weight, output_path + "-weight")
        elif param_name.split("_")[-1] == "bias":
            _SaveWeightBlob2File(weight, output_path + "-bias")
        else:
            print(param_name, "error error")

    elif (
        "bn" in param_name.split("_")[-2]
        or param_name.split("_")[-2] == "sc"
        or param_name.split("_")[-2] == "batchnorm"
    ):
        if param_name.split("_")[-1] == "gamma":
            _SaveWeightBlob2File(weight, output_path + "-gamma")
        elif param_name.split("_")[-1] == "beta":
            _SaveWeightBlob2File(weight, output_path + "-beta")
        else:
            print(param_name, "error error")
    elif "relu" in param_name.split("_")[-2]:
        if param_name.split("_")[-1] == "gamma":
            _SaveWeightBlob2File(weight, output_path + "-alpha")
        else:
            print(param_name, "error error")
    elif param_name.split("_")[-2] == "dt0":
        print("dt0 dt0dt0dt0")
        output_path = of_dump_path + "_".join(param_name.split("_")[0:-2])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if param_name.split("_")[-1] == "weight":
            _SaveWeightBlob2File(weight, output_path + "-weight")
        elif param_name.split("_")[-1] == "bias":
            _SaveWeightBlob2File(weight, output_path + "-bias")
        else:
            print("error error")
    elif (
        param_name.split("_")[-2] == "fc1"
        or param_name.split("_")[-2] == "fc7"
    ):
        if param_name.split("_")[-1] == "weight":
            _SaveWeightBlob2File(weight, output_path + "-weight")
        elif param_name.split("_")[-1] == "bias":
            _SaveWeightBlob2File(weight, output_path + "-bias")
        elif param_name.split("_")[-1] == "beta":
            _SaveWeightBlob2File(weight, output_path + "-beta")
        elif param_name.split("_")[-1] == "gamma":
            _SaveWeightBlob2File(weight, output_path + "-gamma")
        else:
            print(param_name, "error error")
    else:
        print(param_name, "error error")

print("aux_params:")
for param_name in aux_params.keys():
    print(param_name)
    output_path = of_dump_path + "_".join(param_name.split("_")[0:-2])
    weight = aux_params[param_name].asnumpy()
    if (
        "bn" in param_name.split("_")[-3]
        or "fc1" in param_name.split("_")[-3]
        or param_name.split("_")[-3] == "sc"
        or param_name.split("_")[-3] == "batchnorm"
    ):
        if param_name.split("_")[-1] == "mean":
            _SaveWeightBlob2File(weight, output_path + "-moving_mean")
        elif param_name.split("_")[-1] == "var":
            _SaveWeightBlob2File(weight, output_path + "-moving_variance")
        else:
            print(param_name, "error error")
    else:
        print(param_name, "error error")
