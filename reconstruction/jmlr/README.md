## Introduction

JMLR is an efficient high accuracy face reconstruction approach which achieved [Rank-1st](https://tianchi.aliyun.com/competition/entrance/531961/rankingList) of 
[Perspective Projection Based Monocular 3D Face Reconstruction Challenge]([https://tianchi.aliyun.com/competition/entrance/531958/introduction](https://tianchi.aliyun.com/competition/entrance/531961/introduction)) 
of [ECCV-2022 WCPA Workshop](https://sites.google.com/view/wcpa2022).



## Data preparation

1. Download the dataset from WCPA organiser and put it in somewhere.

2. Use ``python rec_builder.py`` to generate cached dataset, which will be used in following steps.
 

## Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=13334 train.py configs/s1.py
```

## Inference Example

```
python inference_simple.py
```

## Results

TODO


