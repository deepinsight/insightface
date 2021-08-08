# Training performance report on NVIDIA RTX3090

[GEFORCE RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/) 
The GeForce RTX™ 3090 is a big ferocious GPU (BFGPU) with TITAN class performance.

Besides, we can also use GeForce RTX™ 3090 to train deep learning models by its FP16 and TF32 supports.



## Test Server Spec

| Key          | Value                                             |
|--------------|---------------------------------------------------|
| CPU          | 2 x Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz |
| Memory       | 384GB                                             |
| GPU          | 8 x GeForce RTX™ 3090                             |
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

| Dataset    | Classes    | Backbone   | Batch-size | FP16 | TF32 | Partial FC | Samples/sec |
|------------|------------|------------|------------|------|------|------------|-------------|
| WebFace40M | 2 Millions | IResNet-50 | 512        | ×    | ×    | ×          | ~1750       |
| WebFace40M | 2 Millions | IResNet-50 | 512        | ×    | √    | ×          | ~1810       |
| WebFace40M | 2 Millions | IResNet-50 | 512        | √    | √    | ×          | ~2056       |
| WebFace40M | 2 Millions | IResNet-50 | 512        | √    | √    | √          | ~2850       |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | √    | √    | ×          | ~2810       |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | √    | √    | √          | ~4220       |
| WebFace40M | 2 Millions | IResNet-50 | 2048       | √    | √    | √          | ~5330       |


### 2. 600K Identities

We use a large dataset which contains about 600k identities to simulate real cases.

| Dataset     | Classes | Backbone   | Batch-size | FP16 | Samples/sec |
|-------------|---------|------------|------------|------|-------------|
| WebFace600K | 618K    | IResNet-50 | 512        | ×    | ~2220       |
| WebFace600K | 618K    | IResNet-50 | 512        | √    | ~2610       |
| WebFace600K | 618K    | IResNet-50 | 1024       | ×    | ~2940       |
| WebFace600K | 618K    | IResNet-50 | 1024       | √    | ~3790       |
| WebFace600K | 618K    | IResNet-50 | 2048       | √    | ~4680       |