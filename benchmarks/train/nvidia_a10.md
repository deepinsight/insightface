# Training performance report on NVIDIA A10

[NVIDIA A10 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) 

We can use A10 to train deep learning models by its FP16 and TF32 supports.



## Test Server Spec

| Key          | Value                                            |
| ------------ | ------------------------------------------------ |
| System       | ServMax G408-X2 Rackmountable Server             |
| CPU          | 2 x Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz    |
| Memory       | 384GB, 12 x Samsung 32GB DDR4-2933               |
| GPU          | 8 x NVIDIA A10 22GB                              |
| Cooling      | 2x Customized GPU Kit for GPU support FAN-1909L2 |
| Hard Drive   | Intel SSD S4500 1.9TB/SATA/TLC/2.5"              |
| OS           | Ubuntu 16.04.7 LTS                               |
| Installation | CUDA 11.1, cuDNN 8.0.5                           |
| Installation | Python 3.7.10                                    |
| Installation | PyTorch 1.9.0 (conda)                            |

This server is donated by [AMAX](https://www.amaxchina.com/), many thanks!



## Experiments on arcface_torch

We report training speed in following table, please also note that:

1. The training dataset is in mxnet record format and located on SSD hard drive.

2. Embedding-size are all set to 512.

3. We use a large dataset which contains about 618K identities to simulate real cases.

| Dataset     | Classes | Backbone    | Batch-size | FP16 | TF32 | Samples/sec |
| ----------- | ------- | ----------- | ---------- | ---- | ---- | ----------- |
| WebFace600K | 618K    | IResNet-50  | 1024       | ×    | ×    | ~2040       |
| WebFace600K | 618K    | IResNet-50  | 1024       | ×    | √    | ~2255       |
| WebFace600K | 618K    | IResNet-50  | 1024       | √    | ×    | ~3300       |
| WebFace600K | 618K    | IResNet-50  | 1024       | √    | √    | ~3360       |
| WebFace600K | 618K    | IResNet-50  | 2048       | √    | √    | ~3940       |
| WebFace600K | 618K    | IResNet-100 | 1024       | √    | √    | ~2210       |
| WebFace600K | 618K    | IResNet-180 | 1024       | √    | √    | ~1410       |


