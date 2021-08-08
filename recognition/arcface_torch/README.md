# Distributed Arcface Training in Pytorch

This is a deep learning library that makes face recognition efficient, and effective, which can train tens of millions
identity on a single server.

## Requirements

- Install [pytorch](http://pytorch.org) (torch>=1.6.0), our doc for [install.md](docs/install.md).
- `pip install -r requirements.txt`.
- Download the dataset
  from [https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
  .

## How to Training

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

## Model Zoo

- The models are available for non-commercial research purposes only.  
- All models can be found in here.  
- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g):   e8pw  
- [onedrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

### Performance on [**ICCV2021-MFR**](http://iccv21-mfr.com/)

ICCV2021-MFR testset consists of non-celebrities so we can ensure that it has very few overlap with public available face 
recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. 
As the result, we can evaluate the FAIR performance for different algorithms.  

For **ICCV2021-MFR-ALL** set, TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6). The 
globalised multi-racial testset contains 242,143 identities and 1,624,305 images. 

For **ICCV2021-MFR-MASK** set, TAR is measured on mask-to-nonmask 1:1 protocal, with FAR less than 0.0001(e-4). 
Mask testset contains 6,964 identities, 6,964 masked images and 13,928 non-masked images. 
There are totally 13,928 positive pairs and 96,983,824 negative pairs.

| Datasets | backbone  | Training throughout | Size / MB  | **ICCV2021-MFR-MASK** | **ICCV2021-MFR-ALL** |
| :---:    | :---      | :---                | :---       |:---                   |:---                  |     
| MS1MV3    | r18  | -              | 91   | **47.85** | **68.33** |
| Glint360k | r18  | 8536           | 91   | **53.32** | **72.07** |
| MS1MV3    | r34  | -              | 130  | **58.72** | **77.36** |
| Glint360k | r34  | 6344           | 130  | **65.10** | **83.02** |
| MS1MV3    | r50  | 5500           | 166  | **63.85** | **80.53** |
| Glint360k | r50  | 5136           | 166  | **70.23** | **87.08** |
| MS1MV3    | r100 | -              | 248  | **69.09** | **84.31** |
| Glint360k | r100 | 3332           | 248  | **75.57** | **90.66** |
| MS1MV3    | mobilefacenet | 12185 | 7.8  | **41.52** | **65.26** |        
| Glint360k | mobilefacenet | 11197 | 7.8  | **44.52** | **66.48** |  

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


## [Speed Benchmark](docs/speed_benchmark.md)

**Arcface Torch** can train large-scale face recognition training set efficiently and quickly. When the number of
classes in training sets is greater than 300K and the training is sufficient, partial fc sampling strategy will get same
accuracy with several times faster training performance and smaller GPU memory. 
Partial FC is a sparse variant of the model parallel architecture for large sacle  face recognition. Partial FC use a 
sparse softmax, where each batch dynamicly sample a subset of class centers for training. In each iteration, only a 
sparse part of the parameters will be updated, which can reduce a lot of GPU memory and calculations. With Partial FC, 
we can scale trainset of 29 millions identities, the largest to date. Partial FC also supports multi-machine distributed 
training and mixed precision training.

![Image text](https://github.com/anxiangsir/insightface_arcface_log/blob/master/partial_fc_v2.png)

More details see 
[speed_benchmark.md](docs/speed_benchmark.md) in docs.

### 1. Training speed of different parallel methods (samples / second), Tesla V100 32GB * 8. (Larger is better)

`-` means training failed because of gpu memory limitations.

| Number of Identities in Dataset | Data Parallel | Model Parallel | Partial FC 0.1 |
| :---    | :--- | :--- | :--- |
|125000   | 4681         | 4824          | 5004     |
|1400000  | **1672**     | 3043          | 4738     |
|5500000  | **-**        | **1389**      | 3975     |
|8000000  | **-**        | **-**         | 3565     |
|16000000 | **-**        | **-**         | 2679     |
|29000000 | **-**        | **-**         | **1855** |

### 2. GPU memory cost of different parallel methods (MB per GPU), Tesla V100 32GB * 8. (Smaller is better)

| Number of Identities in Dataset | Data Parallel | Model Parallel | Partial FC 0.1 |
| :---    | :---      | :---      | :---  |
|125000   | 7358      | 5306      | 4868  |
|1400000  | 32252     | 11178     | 6056  |
|5500000  | **-**     | 32188     | 9854  |
|8000000  | **-**     | **-**     | 12310 |
|16000000 | **-**     | **-**     | 19950 |
|29000000 | **-**     | **-**     | 32324 |

## Evaluation ICCV2021-MFR and IJB-C

More details see [eval.md](docs/eval.md) in docs.

## Test

We tested many versions of PyTorch. Please create an issue if you are having trouble.  

- [x] torch 1.6.0
- [x] torch 1.7.1
- [x] torch 1.8.0
- [x] torch 1.9.0

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
