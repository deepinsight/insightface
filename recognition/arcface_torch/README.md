# Arcface Pytorch (Distributed Version of ArcFace)

## Contents

## Set Up
```shell
torch >= 1.6.0
```  
More details see [eval.md](docs/install.md) in docs.

## Training
### 1. Single node, 8 GPUs:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py
```
### 2. Multiple nodes, each node 8 GPUs:  
Node 0:  
```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=1234 train.py
```
Node 1:  
```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=1234 train.py
```

### 3.Training resnet2060 with 8 GPUs:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --network r2060
```

## Speed Benchmark
![Image text](https://github.com/nttstar/insightface-resources/blob/master/images/arcface_speed.png)

ArcFace_torch can train large-scale face recognition training set efficiently and quickly.  
When the number of classes in training sets is greater than 300K and the training is sufficient, 
partial fc sampling strategy will get same accuracy with several times faster training performance and smaller GPU memory.

1. Training speed of different parallel methods (samples/second), Tesla V100 32GB * 8. (Larger is better)

| Method                 | Bs1024-R100-2 Million Identities | Bs1024-R50-4 Million Identities | Bs512-R50-8 Million Identities |
| :---                   |    :---                          | :---                            | :---                     |
| Data Parallel          |    1                             | 1                               | 1                        |
| Model Parallel         |    1362                          | 1600                            | 482                      |
| Fp16 + Model Parallel  |    2006                          | 2165                            | 767                      | 
| Fp16 + Partial Fc 0.1  |    3247                          | 4385                            | 3001                     | 

2. GPU memory cost of different parallel methods (GB per GPU), Tesla V100 32GB * 8. (Smaller is better)

| Method                 | Bs1024-R100-2 Million Identities   | Bs1024-R50-4 Million Identities   | Bs512-R50-8 Million Identities |
| :---                   |    :---                           | :---                             | :---                     |
| Data Parallel          |    OOM                            | OOM                              | OOM                      |
| Model Parallel         |    27.3                           | 30.3                             | 32.1                      |
| Fp16 + Model Parallel  |    20.3                           | 26.6                             | 32.1                      | 
| Fp16 + Partial Fc 0.1  |    11.9                           | 10.8                             | 11.1                      | 


## Evaluation IJBC
More details see [eval.md](docs/eval.md) in docs.

## Inference
```shell
python inference.py --weight ms1mv3_arcface_r50/backbone.pth --network r50
```

## Model Zoo  

The models are available for non-commercial research purposes only.

All Model Can be found in here.  
[Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g):   e8pw  
[onedrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

### MS1MV3

`config:` batchsize is 128 * 8, using mixed precision training(pytorch1.6+), loss is arcface.  


|   Datasets          |    log     | backbone    | IJBC(1e-05) | IJBC(1e-04) |agedb30|cfp_fp|lfw  | 
| :---:               |    :---    | :---        | :---        | :---        |:---   |:---  |:--- |  
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r18_fp16/training.log)  | r18-fp16      | 92.07 | 94.66 | 97.77 | 97.73 | 99.77 |
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r34_fp16/training.log)  | r34-fp16      | 94.10 | 95.90 | 98.10 | 98.67 | 99.80 |
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r50_fp16/training.log)  | r50-fp16      | 94.79 | 96.46 | 98.35 | 98.96 | 99.83 | 
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r100_fp16/training.log) | r100-fp16     | 95.31 | 96.81 | 98.48 | 99.06 | 99.85 | 
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r2060_fp16/training.log)| **r2060-fp16**| 95.34 | 97.11 | 98.67 | 99.24 | 99.87 |                 
   
### Glint360k  


`config:` batchsize is 128 * 8, using mixed precision training(pytorch1.6+) and partial fc sampling(0.1), loss is cosface.  

|   Datasets          | log   |backbone               | IJBC(1e-05) | IJBC(1e-04) |agedb30|cfp_fp|lfw  | 
| :---:               | :---  |:---                   | :---        | :---        |:---   |:---  |:--- |
| Glint360k-Cosface   |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r18_fp16_0.1/training.log) |r18-fp16-0.1  | 93.16 | 95.33 | 97.72 | 97.73 | 99.77 |
| Glint360k-Cosface   |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r34_fp16_0.1/training.log) |r34-fp16-0.1  | 95.16 | 96.56 | 98.33 | 98.78 | 99.82 |
| Glint360k-Cosface   |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r50_fp16_0.1/training.log) |r50-fp16-0.1  | 95.61 | 96.97 | 98.38 | 99.20 | 99.83 |
| Glint360k-Cosface   |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r100_fp16_0.1/training.log)|r100-fp16-0.1 | 95.88 | 97.32 | 98.48 | 99.29 | 99.82 |

More details see [eval.md](docs/modelzoo.md) in docs.


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
