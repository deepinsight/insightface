## Introduction

JMLR is an efficient high accuracy face reconstruction approach which achieved [Rank-1st](https://tianchi.aliyun.com/competition/entrance/531961/rankingList) of 
[Perspective Projection Based Monocular 3D Face Reconstruction Challenge](https://tianchi.aliyun.com/competition/entrance/531961/introduction) 
of [ECCV-2022 WCPA Workshop](https://sites.google.com/view/wcpa2022).

Paper in [arXiv](https://arxiv.org/abs/2208.07142).


## Method Pipeline


<img src="https://github.com/nttstar/insightface-resources/blob/master/images/jmlr_pipeline.jpg?raw=true" width="800" alt="jmlr-pipeline"/>


## Data preparation

1. Download the dataset from WCPA organiser and put it at somewhere.

2. Create `cache_align/` dir and put `flip_index.npy` file under it.

3. Check `configs/s1.py` and fix the location to yours.

4. Use ``python rec_builder.py`` to generate cached dataset, which will be used in following steps.
 

## Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=13334 train.py configs/s1.py
```

## Inference Example

```
python inference_simple.py
```

## Resources

[flip_index.npy](https://drive.google.com/file/d/1fZ4cRyvQeehwKoMKKSmXUmTx5GEJwyrT/view?usp=sharing)

[pretrained-model](https://drive.google.com/file/d/1qSpqDDLQfcPeFr2b82IZrK8QC_3lci3l/view?usp=sharing)

[projection_matrix.txt](https://drive.google.com/file/d/1joiu-V0qEZxil_AHcg_W726nRxE8Q4dm/view?usp=sharing)

## Results

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/jmlr_id.jpg?raw=true" width="800" alt="jmlr-id"/>

