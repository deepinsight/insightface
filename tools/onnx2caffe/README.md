# Convert ONNX to Caffe

This tool is modified from [onnx2caffe](https://github.com/MTlab/onnx2caffe) by MTlab.

We added some OPs to support one-stage mmdetection models.

### Dependencies
* pycaffe (with builtin Upsample and Permute layers)
* onnx  


### How to use
To convert onnx model to caffe:
```
python convertCaffe.py ./model/mmdet.onnx ./model/a.prototxt ./model/a.caffemodel
```

### Current support operation
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
* **Permute**
* **Scale**

