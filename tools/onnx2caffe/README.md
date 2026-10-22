# Convert ONNX to Caffe

This tool is modified from [onnx2caffe](https://github.com/MTlab/onnx2caffe) by MTlab.

We added some OPs to support one-stage mmdetection models.

### Dependencies

* pycaffe (with builtin Upsample and Permute layers)
* onnx
* numpy


### How to use

Run [convertCaffe.py](convertCaffe.py) from this directory using an environment
with the dependencies above. The three positional arguments are the input ONNX
file, the output Caffe network definition, and the output Caffe weights:

```bash
python convertCaffe.py ./model/mmdet.onnx ./model/a.prototxt ./model/a.caffemodel
```

The `./model/` paths are examples: provide your own input model and create the
output parent directory before running the command. The script prints the model
output names when conversion completes.

### Registered ONNX operations

The [converter registry](onnx2caffe/_operators.py) handles the operations below.
Supported attributes and tensor shapes depend on each converter implementation.

* Conv
* ConvTranspose
* BatchNormalization
* MaxPool
* AveragePool
* Relu
* Sigmoid
* Dropout
* Gemm (InnerProduct only)
* Add
* Mul
* Reshape
* Upsample
* Concat
* Flatten
* **Resize**
* **Transpose**
* **Softmax**

`Transpose` maps to a Caffe `Permute` layer. `Scale` is a Caffe layer emitted by
the `BatchNormalization` and `Mul` converters, rather than an ONNX operation name.

For direct ONNX inference, see the [InsightFace Python library](../../python-package/README.md).
