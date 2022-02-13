# Training performance report on NVIDIA® V100

[NVIDIA® V100](https://www.nvidia.com/en-us/data-center/v100/) 
NVIDIA® V100 Tensor Core is the most advanced data center GPU ever built to accelerate AI, high performance computing (HPC), data science and graphics. It’s powered by NVIDIA Volta architecture, comes in 16 and 32GB configurations, and offers the performance of up to 32 CPUs in a single GPU.

Besides, we can also use NVIDIA® V100 to train deep learning models by its FP16 and FP32 supports.

## Test Server Spec

| Key          | Value                                        |
|--------------|----------------------------------------------|
| CPU          | 2 x Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz |
| Memory       | 384GB                                        |
| GPU          | 8 x Tesla V100-SXM2-32GB                     |
| OS           | Ubuntu 16.04 LTS                             |
| Installation | CUDA 10.2                                    |
| Installation | Python 3.7.3                                 |
| Installation | PyTorch 1.9.0 (pip)                          |

## Experiments on arcface_torch

We report training speed in following table, please also note that:

1. The training dataset is SyntheticDataset.

2. Embedding-size are all set to 512.

### 1. 2 Million Identities

We use a large dataset which contains about 2 millions identities to simulate real cases.

| Dataset    | Classes    | Backbone   | Batch-size | FP16 | Partial FC | Samples/sec |
|------------|------------|------------|------------|------|------------|-------------|
| WebFace40M | 2 Millions | IResNet-50 | 512        | ×    | ×          | ~1868       |
| WebFace40M | 2 Millions | IResNet-50 | 512        | x    | √          | ~2712       |
| WebFace40M | 2 Millions | IResNet-50 | 512        | √    | ×          | ~2576       |
| WebFace40M | 2 Millions | IResNet-50 | 512        | √    | √          | ~4501       |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | ×    | ×          | ~1960       |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | x    | √          | ~2922       |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | √    | ×          | ~2810       |
| WebFace40M | 2 Millions | IResNet-50 | 1024       | √    | √          | ~5430       |
| WebFace40M | 2 Millions | IResNet-50 | 2048       | √    | √          | ~6095       |

### 2. 600K Identities

We use a large dataset which contains about 600k identities to simulate real cases.

| Dataset     | Classes | Backbone   | Batch-size | FP16 | Samples/sec |
|-------------|---------|------------|------------|------|-------------|
| WebFace600K | 618K    | IResNet-50 | 512        | ×    | ~2430       |
| WebFace600K | 618K    | IResNet-50 | 512        | √    | ~3889       |
| WebFace600K | 618K    | IResNet-50 | 1024       | ×    | ~2607       |
| WebFace600K | 618K    | IResNet-50 | 1024       | √    | ~4322       |
| WebFace600K | 618K    | IResNet-50 | 2048       | √    | ~4921       |
