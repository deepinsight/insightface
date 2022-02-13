# InsightFace Edge Inference and Deployment

In this tutorial, we give examples and benchmarks of running insightface models on edge devices, mainly using 8-bits quantization technologies to make acceleration.

## Recognition

In recognition tutorial, we use an open-source model: *IR50@Glint360K*, and use a hard private 1:N testset(N=50000). The metric contains Rank1 and TAR@FAR<=e-3.



Granularity and symmetry both stand for quantization setting, and mostly defined by hardware providers. Symmetric uses INT8 to save quantization results while Asymmetric uses UINT8 type.

| Hardware    | Provider | Type | Backend     | Time | Granularity | Symmetry   | Rank1-Acc | TAR@FAR<=e-3 |
| ----------- | -------- | ---- | ----------- | ---- | ----------- | ---------- | --------- | ------------ |
| V100        | NVIDIA   | GPU  | onnxruntime | 4ms  | -           | -          | 80.94     | 30.77        |
| Jetson NX   | NVIDIA   | GPU  | TensorRT    | 16ms | Per-channel | Symmetric  | 79.26     | 31.07        |
| A311D       | Khadas   | ASIC | Tengine     | 26ms | Per-tensor  | Asymmetric | 77.83     | 26.58        |
| A311D*      | Khadas   | ASIC | Tengine     | 26ms | Per-tensor  | Asymmetric | 79.38     | 28.59        |
| NXP-IMX8P   | NXP      | ASIC | Tengine     | 24ms | Per-tensor  | Asymmetric | 77.87     | 26.80        |
| NXP-IMX8P*  | NXP      | ASIC | Tengine     | 24ms | Per-tensor  | Asymmetric | 79.42     | 28.39        |
| RV1126      | Rockchip | ASIC | RKNN        | 38ms | Per-tensor  | Asymmetric | 75.60     | 24.23        |
| RV1126*     | Rockchip | ASIC | RKNN        | 38ms | Per-tensor  | Asymmetric | 77.82     | 26.30        |

Suffix-* means mixed mode: using float32 model for gallery while using quantized model for probe images. Result features are all in float32 type.

The example code of running quantized networks can be now found at [Tengine](https://github.com/OAID/Tengine/tree/tengine-lite/demos). Later, we will put a copy here and give full tutorial on how to quantize recognition models from 0 to 1.



## Detection

TODO

