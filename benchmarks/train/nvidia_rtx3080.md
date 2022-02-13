# Training performance report on NVIDIA RTX3080

[GeForce RTX 3080](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/) 
The GeForce RTX™ 3080 Ti and RTX 3080 graphics cards deliver the ultra performance that gamers crave, powered by Ampere—NVIDIA’s 2nd gen RTX architecture. They are built with enhanced RT Cores and Tensor Cores, new streaming multiprocessors, and superfast G6X memory for an amazing gaming experience.

Besides, we can also use GeForce RTX™ 3080 to train deep learning models by its FP16 and TF32 supports.



## Test Server Spec

| Key          | Value                                             |
|--------------|---------------------------------------------------|
| CPU          | 2 x Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz |
| Memory       | 384GB                                             |
| GPU          | 8 x GeForce RTX™ 3080                             |
| OS           | Ubuntu 18.04.4 LTS                                |
| Installation | CUDA 11.1,                                        |
| Installation | Python 3.7.3                                      |
| Installation | PyTorch 1.9.0 (pip)                               |


## Experiments on arcface_torch

We report training speed in following table, please also note that:

1. The training dataset is SyntheticDataset.

2. Embedding-size are all set to 512.


### 1. 2 Million Identities

We use a large dataset which contains about 2 millions identities to simulate real cases.


| Dataset    | Classes    | Backbone   | Batch-size | FP16 | Partial FC | Samples/sec |
|------------|------------|------------|------------|------|------------|-------------|
| WebFace40M | 2 Millions | IResNet-50 | 512        | ×    | ×          | Fail        |
| WebFace40M | 2 Millions | IResNet-50 | 512        | x    | √          | ~2190       |
| WebFace40M | 2 Millions | IResNet-50 | 512        | √    | ×          | Fail        |
| WebFace40M | 2 Millions | IResNet-50 | 512        | √    | √          | ~2620       |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | ×    | ×          | Fail        |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | x    | √          | Fail        |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | √    | ×          | Fail        |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | √    | √          | ~3800       |

### 2. 600K Identities

We use a large dataset which contains about 600k identities to simulate real cases.

| Dataset     | Classes | Backbone   | Batch-size | Partial FC | FP16 | Samples/sec |
|-------------|---------|------------|------------|------------|------|-------------|
| WebFace600K | 618K    | IResNet-50 | 512        | ×          | ×    | ~2023       |
| WebFace600K | 618K    | IResNet-50 | 512        | ×          | √    | ~2392       |
| WebFace600K | 618K    | IResNet-50 | 1024       | ×          | ×    | Fail        |
| WebFace600K | 618K    | IResNet-50 | 1024       | ×          | √    | Fail        |
| WebFace600K | 618K    | IResNet-50 | 1024       | √          | √    | ~4010       |
