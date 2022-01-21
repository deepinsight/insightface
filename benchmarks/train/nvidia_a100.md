# Training performance report on NVIDIA A100

[NVIDIA A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/) 



## Test Server Spec

| Key          | Value                                            |
| ------------ | ------------------------------------------------ |
| System       | ServMax G408-X2 Rackmountable Server             |
| CPU          | 2 x Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz    |
| Memory       | 384GB, 12 x Samsung 32GB DDR4-2933               |
| GPU          | 8 x NVIDIA A100 80GB                             |
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
3. We use large datasets with about 618K/2M identities to simulate real cases.
4. We test the 10K batch-size on real dataset to take the full advantage of 80GB memory.
5. We also test on huge synthetic datasets which include 50M~80M classes.

| Dataset     | Classes | Backbone    | Batch-size | PFC  | FP16 | TF32 | Samples/sec | GPU Mem(GB) |
| ----------- | ------- | ----------- | ---------- | ---- | ---- | ---- | ----------- | ----------- |
| WebFace600K | 618K    | IResNet-50  | 1024       | ×    | ×    | ×    | ~3670       | ~18.2       |
| WebFace600K | 618K    | IResNet-50  | 1024       | ×    | ×    | √    | ~4760       | ~15.0       |
| WebFace600K | 618K    | IResNet-50  | 1024       | ×    | √    | ×    | ~5170       | ~10.1       |
| WebFace600K | 618K    | IResNet-50  | 1024       | ×    | √    | √    | ~5400       | ~10.1       |
| WebFace600K | 618K    | IResNet-50  | 2048       | ×    | √    | √    | ~7780       | ~16.4       |
| WebFace600K | 618K    | IResNet-50  | 10240      | ×    | √    | √    | ~9400       | ~66.7       |
| WebFace600K | 618K    | IResNet-100 | 1024       | ×    | √    | √    | ~3700       | ~13.1       |
| WebFace600K | 618K    | IResNet-180 | 1024       | ×    | √    | √    | ~2380       | ~17.5       |
| WebFace2M   | 2M      | IResNet-100 | 1024       | ×    | √    | √    | ~3480       | ~20.5       |
| WebFace2M   | 2M      | IResNet-180 | 1024       | ×    | √    | √    | ~2350       | ~25.0       |
| WebFace2M   | 2M      | IResNet-300 | 1024       | ×    | √    | √    | ~1541       | ~32.6       |
| Virtual     | 50M     | IResNet-50  | 1024       | 0.1  | √    | √    | ~2700       | ~54.1       |
| Virtual     | 70M     | IResNet-50  | 1024       | 0.1  | √    | √    | ~2170       | ~73.7       |
| Virtual     | 80M     | IResNet-50  | 1024       | 0.1  | √    | √    | ~1080       | ~79.6       |


