# Distributed Arcface Training in Pytorch

This is a deep learning library that makes face recognition efficient, and effective, which can train tens of millions
identity on a single server.

## Requirements

- Install [PyTorch](http://pytorch.org) (torch>=1.6.0), our doc for [install.md](docs/install.md).
- (Optional) Install [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/), our doc for [install_dali.md](docs/install_dali.md).
- `pip install -r requirements.txt`.
  
## How to Training

To train a model, run `train.py` with the path to the configs.  
The example commands below show how to run
distributed training.

### 1. To run on a machine with 8 GPUs:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/ms1mv3_r50_lr02
```

### 2. To run on 2 machines with 8 GPUs each:

Node 0:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 train.py configs/webface42m_r100_lr01_pfc02_bs4k_16gpus
```

Node 1:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 train.py configs/webface42m_r100_lr01_pfc02_bs4k_16gpus
```

## Download Datasets or Prepare Datasets

- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images)
- [WebFace42M](docs/prepare_webface42m.md) (2M IDs, 42.5M images)

## Model Zoo

- The models are available for non-commercial research purposes only.  
- All models can be found in here.  
- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g): e8pw  
- [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

### Performance on IJB-C and [**ICCV2021-MFR**](https://github.com/deepinsight/insightface/blob/master/challenges/mfr/README.md)

ICCV2021-MFR testset consists of non-celebrities so we can ensure that it has very few overlap with public available face 
recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. 
As the result, we can evaluate the FAIR performance for different algorithms.  

For **ICCV2021-MFR-ALL** set, TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6). The 
globalised multi-racial testset contains 242,143 identities and 1,624,305 images. 



| Datasets                 | Backbone   | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | Training Throughout | log                                                                                                                                                                                                         |
|:-------------------------|:-----------|:------------|:------------|:------------|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MS1MV3                   | mobileface | 65.76       | 94.44       | 91.85       | ~13000              | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_mobileface_lr02/training.log)\|[config](configs/ms1mv3_mobileface_lr02.py)                                         |
| Glint360K                | mobileface | 69.83       | 95.17       | 92.58       | -11000              | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_mobileface_lr02_bs4k/training.log)\|[config](configs/glint360k_mobileface_lr02_bs4k.py)                         |
| WebFace42M-PartialFC-0.2 | mobileface | 73.80       | 95.40       | 92.64       | (16GPUs)~18583      | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_mobilefacenet_pfc02_bs8k_16gpus/training.log)\|[config](configs/webface42m_mobilefacenet_pfc02_bs8k_16gpus.py) |
| MS1MV3                   | r100       | 83.23       | 96.88       | 95.31       | ~3400               | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r100_lr02/training.log)\|[config](configs/ms1mv3_r100_lr02.py)                                                     |
| Glint360K                | r100       | 90.86       | 97.53       | 96.43       | ~5000               | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r100_lr02_bs4k_16gpus/training.log)\|[config](configs/glint360k_r100_lr02_bs4k_16gpus.py)                                                                                                                                                                |
| WebFace42M-PartialFC-0.2 | r50(bs4k)  | 93.83       | 97.53       | 96.16       | (8 GPUs)~5900       | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_bs4k_pfc02/training.log)\|[config](configs/webface42m_r50_lr01_pfc02_bs4k_8gpus.py)                        |
| WebFace42M-PartialFC-0.2 | r50(bs8k)  | 93.96       | 97.46       | 96.12       | (16GPUs)~11000      | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_lr01_pfc02_bs8k_16gpus/training.log)\|[config](configs/webface42m_r50_lr01_pfc02_bs8k_16gpus.py)           |
| WebFace42M-PartialFC-0.2 | r50(bs4k)  | 94.04       | 97.48       | 95.94       | (32GPUs)~17000      | log\|[config](configs/webface42m_r50_lr01_pfc02_bs4k_32gpus.py)                                                                                                                                             |
| WebFace42M-PartialFC-0.2 | r100(bs4k) | 96.69       | 97.85       | 96.63       | (16GPUs)~5200       | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r100_bs4k_pfc02/training.log)\|[config](configs/webface42m_r100_lr01_pfc02_bs4k_16gpus.py)                     |
| WebFace42M-PartialFC-0.2 | r200       | -           | -           | -           | -                   | log\|config                                                                                                                                                                                                 |

`PartialFC-0.2` means negivate class centers sample rate is 0.2.


## Speed Benchmark

`arcface_torch` can train large-scale face recognition training set efficiently and quickly. When the number of
classes in training sets is greater than 1 Million, partial fc sampling strategy will get same
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
|:--------------------------------|:--------------|:---------------|:---------------|
| 125000                          | 4681          | 4824           | 5004           |
| 1400000                         | **1672**      | 3043           | 4738           |
| 5500000                         | **-**         | **1389**       | 3975           |
| 8000000                         | **-**         | **-**          | 3565           |
| 16000000                        | **-**         | **-**          | 2679           |
| 29000000                        | **-**         | **-**          | **1855**       |

### 2. GPU memory cost of different parallel methods (MB per GPU), Tesla V100 32GB * 8. (Smaller is better)

| Number of Identities in Dataset | Data Parallel | Model Parallel | Partial FC 0.1 |
|:--------------------------------|:--------------|:---------------|:---------------|
| 125000                          | 7358          | 5306           | 4868           |
| 1400000                         | 32252         | 11178          | 6056           |
| 5500000                         | **-**         | 32188          | 9854           |
| 8000000                         | **-**         | **-**          | 12310          |
| 16000000                        | **-**         | **-**          | 19950          |
| 29000000                        | **-**         | **-**          | 32324          |


## Citations

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
