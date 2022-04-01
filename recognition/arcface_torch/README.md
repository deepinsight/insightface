# Distributed Arcface Training in Pytorch

This is a deep learning library that makes face recognition efficient, and effective, which can train tens of millions
identity on a single server.

## Requirements

In order to enjoy the features of the new torch, we have upgraded the torch to 1.9.0.
torch version before than 1.9.0 may not work in the future.

- Install [PyTorch](http://pytorch.org) (torch>=1.9.0), our doc for [install.md](docs/install.md).
- (Optional) Install [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/), our doc for [install_dali.md](docs/install_dali.md).
- `pip install -r requirement.txt`.
  
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


> 1. Large Scale Datasets

| Datasets         | Backbone    | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | Training Throughout | log                                                                                                                                             |
|:-----------------|:------------|:------------|:------------|:------------|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| MS1MV3           | mobileface  | 65.76       | 94.44       | 91.85       | ~13000              | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_mobileface_lr02/training.log)                     |
| Glint360K        | mobileface  | 69.83       | 95.17       | 92.58       | -11000              | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_mobileface_lr02_bs4k/training.log)             |
| WF42M-PFC-0.2    | mobileface  | 73.80       | 95.40       | 92.64       | (16GPUs)~18583      | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_mobilefacenet_pfc02_bs8k_16gpus/training.log) |
| MS1MV3           | r100        | 83.23       | 96.88       | 95.31       | ~3400               | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r100_lr02/training.log)                           |
| Glint360K        | r100        | 90.86       | 97.53       | 96.43       | ~5000               | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r100_lr02_bs4k_16gpus/training.log)            |
| WF42M-PFC-0.2    | r50(bs4k)   | 93.83       | 97.53       | 96.16       | (8 GPUs)~5900       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_bs4k_pfc02/training.log)                  |
| WF42M-PFC-0.2    | r50(bs8k)   | 93.96       | 97.46       | 96.12       | (16GPUs)~11000      | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_lr01_pfc02_bs8k_16gpus/training.log)      |
| WF42M-PFC-0.2    | r50(bs4k)   | 94.04       | 97.48       | 95.94       | (32GPUs)~17000      | click me                                                                                                                                        |
| WF42M-PFC-0.0018 | r100(bs16k) | 93.08       | 97.51       | 95.88       | (32GPUs)~10000      | click me                                                                                                                                        |
| WF42M-PFC-0.2    | r100(bs4k)  | 96.69       | 97.85       | 96.63       | (16GPUs)~5200       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r100_bs4k_pfc02/training.log)                 |

> 2. VIT For Face Recognition

| Datasets      | Backbone     | FLOPs | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | Training Throughout | log      |
|:--------------|:-------------|:------|:------------|:------------|:------------|:--------------------|:---------|
| WF42M-PFC-0.3 | R18(bs4k)    | 2.6   | 79.13       | 95.77       | 93.36       | -                   | click me |
| WF42M-PFC-0.3 | R50(bs4k)    | 6.3   | 94.03       | 97.48       | 95.94       | -                   | click me |
| WF42M-PFC-0.3 | R100(bs4k)   | 12.1  | 96.69       | 97.82       | 96.45       | -                   | click me |
| WF42M-PFC-0.3 | R200(bs4k)   | 23.5  | 97.70       | 97.97       | 96.93       | -                   | click me |
| WF42M-PFC-0.3 | VIT-T(bs24k) | 1.5   | 92.24       | 97.31       | 95.97       | (64GPUs)~35000      | click me |
| WF42M-PFC-0.3 | VIT-S(bs24k) | 5.7   | 95.87       | 97.73       | 96.57       | (64GPUs)~25000      | click me |
| WF42M-PFC-0.3 | VIT-B(bs24k) | 11.4  | 97.42       | 97.90       | 97.04       | (64GPUs)~13800      | click me |
| WF42M-PFC-0.3 | VIT-L(bs24k) | 25.3  | 97.85       | 98.00       | 97.23       | (64GPUs)~9406       | click me |

WF42M means WebFace42M, `PFC-0.3` means negivate class centers sample rate is 0.3.

> 3. Noisy Datasets
  
| Datasets                 | Backbone | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | log      |
|:-------------------------|:---------|:------------|:------------|:------------|:---------|
| WF12M-Flip(40%)          | R50      | 43.87       | 88.35       | 80.78       | click me |
| WF12M-Flip(40%)-PFC-0.3* | R50      | 80.20       | 96.11       | 93.79       | click me |
| WF12M-Conflict           | R50      | 79.93       | 95.30       | 91.56       | click me |
| WF12M-Conflict-PFC-0.3*  | R50      | 91.68       | 97.28       | 95.75       | click me |

WF12M means WebFace12M, `+PFC-0.3*` denotes additional abnormal inter-class filtering.




## Speed Benchmark
<div><img src="https://github.com/anxiangsir/insightface_arcface_log/blob/master/pfc_exp.png" width = "90%" /></div>


**Arcface-Torch** can train large-scale face recognition training set efficiently and quickly. When the number of
classes in training sets is greater than 1 Million, partial fc sampling strategy will get same
accuracy with several times faster training performance and smaller GPU memory. 
Partial FC is a sparse variant of the model parallel architecture for large sacle  face recognition. Partial FC use a 
sparse softmax, where each batch dynamicly sample a subset of class centers for training. In each iteration, only a 
sparse part of the parameters will be updated, which can reduce a lot of GPU memory and calculations. With Partial FC, 
we can scale trainset of 29 millions identities, the largest to date. Partial FC also supports multi-machine distributed 
training and mixed precision training.



More details see 
[speed_benchmark.md](docs/speed_benchmark.md) in docs.

> 1. Training speed of different parallel methods (samples / second), Tesla V100 32GB * 8. (Larger is better)

`-` means training failed because of gpu memory limitations.

| Number of Identities in Dataset | Data Parallel | Model Parallel | Partial FC 0.1 |
|:--------------------------------|:--------------|:---------------|:---------------|
| 125000                          | 4681          | 4824           | 5004           |
| 1400000                         | **1672**      | 3043           | 4738           |
| 5500000                         | **-**         | **1389**       | 3975           |
| 8000000                         | **-**         | **-**          | 3565           |
| 16000000                        | **-**         | **-**          | 2679           |
| 29000000                        | **-**         | **-**          | **1855**       |

> 2. GPU memory cost of different parallel methods (MB per GPU), Tesla V100 32GB * 8. (Smaller is better)

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
@inproceedings{an2022pfc,
  title={Killing Two Birds with One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
  author={An, Xiang and Deng, Jiangkang and Guo, Jia and Feng, Ziyong and Zhu, Xuhan and Jing, Yang and Tongliang, Liu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
@inproceedings{zhu2021webface260m,
  title={Webface260m: A benchmark unveiling the power of million-scale deep face recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10492--10502},
  year={2021}
}
```
