# Distributed Arcface training in Pytorch

This is a deep learning library that makes face recognition efficient, and effective, which can train tens of millions
identity on a single server.

## Requirements

- Install [pytorch](http://pytorch.org) (1.6.0 <= troch < 1.9.0), our doc for [install.md](docs/install.md).
- `pip install -r requirements.txt`.
- Download the dataset
  from [https://github.com/deepinsight/insightface/wiki/Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
  .

## Training

To train a model, run `train.py` with the path to the configs:

### 1. Single node, 8 GPUs:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/ms1mv3_r50
```

### 2. Multiple nodes, each node 8 GPUs:

Node 0:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=1234 train.py train.py configs/ms1mv3_r50
```

Node 1:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=1234 train.py train.py configs/ms1mv3_r50
```

### 3.Training resnet2060 with 8 GPUs:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/ms1mv3_r2060.py
```

## Speed Benchmark

![Image text](https://github.com/nttstar/insightface-resources/blob/master/images/arcface_speed.png)

**Arcface Torch** can train large-scale face recognition training set efficiently and quickly. When the number of
classes in training sets is greater than 300K and the training is sufficient, partial fc sampling strategy will get 
same accuracy with several times faster training performance and smaller GPU memory.

1. Training speed of different parallel methods (samples/second), Tesla V100 32GB * 8. (Larger is better)

| Method                 | Bs1024-R100-2 Million Identities | Bs1024-R50-4 Million Identities | Bs512-R50-8 Million Identities |
| :---                   |    :---                          | :---                            | :---          |
| Data Parallel          |    1                             | 1                               | 1             |
| Model Parallel         |    1362                          | 1600                            | 482           |
| Fp16 + Model Parallel  |    2006                          | 2165                            | 767           | 
| Fp16 + Partial Fc 0.1  |    3247                          | 4385                            | 3001          | 

2. GPU memory cost of different parallel methods (GB per GPU), Tesla V100 32GB * 8. (Smaller is better)

| Method                 | Bs1024-R100-2 Million Identities   | Bs1024-R50-4 Million Identities   | Bs512-R50-8 Million Identities |
| :---                   |    :---                           | :---                             | :---        |
| Data Parallel          |    OOM                            | OOM                              | OOM         |
| Model Parallel         |    27.3                           | 30.3                             | 32.1        |
| Fp16 + Model Parallel  |    20.3                           | 26.6                             | 32.1        | 
| Fp16 + Partial Fc 0.1  |    11.9                           | 10.8                             | 11.1        |

## Model Zoo

- The models are available for non-commercial research purposes only.  
- All models can be found in here.  
- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g):   e8pw  
- [onedrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

### Performance on [**ICCV2021-MFR**](http://iccv21-mfr.com/)

| Datasets | backbone  | Training throughout | Size / MB  | **ICCV2021-MFR-MASK** | **ICCV2021-MFR-ALL** |
| :---:    | :---      | :---                | :---       |:---                   |:---                  |    
| MS1MV3    | mobilefacenet | 12185 | 4.5  | **36.12** | **59.78** |        
| MS1MV3    | r18  | -              | 91   | **47.85** | **68.33** |
| Glint360k | r18  | -              | 91   | **53.32** | **72.07** |
| MS1MV3    | r34  | -              | 130  | **58.72** | **77.36** |
| Glint360k | r34  | -              | 130  | **65.10** | **83.02** |
| MS1MV3    | r50  | 5500           | 166  | **63.85** | **80.53** |
| Glint360k | r50  | -              | 166  | **70.23** | **87.08** |
| MS1MV3    | r100 | -              | 248  | **69.09** | **84.31** |
| Glint360k | r100 | -              | 248  | **75.57** | **90.66** |

### Performance on IJB-C and Verification Datasets

|   Datasets | backbone      | IJBC(1e-05) | IJBC(1e-04) | agedb30 | cfp_fp | lfw  |  log    |
| :---:      |    :---       | :---          | :---  | :---  |:---   |:---    |:---     |  
| MS1MV3     | r18      | 92.07 | 94.66 | 97.77 | 97.73 | 99.77 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r18_fp16/training.log)|         
| MS1MV3     | r34      | 94.10 | 95.90 | 98.10 | 98.67 | 99.80 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r34_fp16/training.log)|        
| MS1MV3     | r50      | 94.79 | 96.46 | 98.35 | 98.96 | 99.83 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r50_fp16/training.log)|         
| MS1MV3     | r100     | 95.31 | 96.81 | 98.48 | 99.06 | 99.85 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r100_fp16/training.log)|        
| MS1MV3     | **r2060**| 95.34 | 97.11 | 98.67 | 99.24 | 99.87 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r2060_fp16/training.log)|
| Glint360k  |r18-0.1   | 93.16 | 95.33 | 97.72 | 97.73 | 99.77 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r18_fp16_0.1/training.log)| 
| Glint360k  |r34-0.1   | 95.16 | 96.56 | 98.33 | 98.78 | 99.82 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r34_fp16_0.1/training.log)| 
| Glint360k  |r50-0.1   | 95.61 | 96.97 | 98.38 | 99.20 | 99.83 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r50_fp16_0.1/training.log)| 
| Glint360k  |r100-0.1  | 95.88 | 97.32 | 98.48 | 99.29 | 99.82 |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r100_fp16_0.1/training.log)|

[comment]: <> (More details see [model.md]&#40;docs/modelzoo.md&#41; in docs.)

## Evaluation IJB-C, ICCV2021-MFR

More details see [eval.md](docs/eval.md) in docs.

## Test

We test on PyTorch versions 1.6.0, 1.7.1, and 1.8.0. Please create an issue if you are having trouble.

## Citation

```
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
@inproceedings{an2020partical_fc,
  title={Partial FC: Training 10 Million Identities on a Single Machine},
  author={An, Xiang and Zhu, Xuhan and Xiao, Yang and Wu, Lan and Zhang, Ming and Gao, Yuan and Qin, Bin and
  Zhang, Debing and Fu Ying},
  booktitle={Arxiv 2010.05222},
  year={2020}
}
```
