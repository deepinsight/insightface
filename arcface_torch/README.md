# Distributed ScaleFace Training in Pytorch

This is a fork of [ArcFace repository](https://github.com/deepinsight/insightface)

Here we improve Arcface training pipeline to train both uncertainty embedder and uncertainty head

## Requirements

- Install [pytorch](http://pytorch.org) (torch>=1.6.0), our doc for [install.md](docs/install.md).
- `pip install -r requirements.txt`.
- Download the dataset
  from [https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
  .

## How to Training

To train a model, run `train.py` with the path to the configs. Configs, describing different experiments are stored ```configs``` folder:

### Single node, 4 GPUs:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/02_sigm_mul_coef_selection/coef_32.py
```

## Model Zoo

- The models are available for non-commercial research purposes only.  
- All models can be found in here.  
- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g):   e8pw  
- [onedrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)
