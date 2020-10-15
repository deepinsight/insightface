## Partial-FC
Partial FC is a distributed deep learning training framework for face recognition. The goal of Partial FC is to facilitate large-scale classification task (e.g. 10 or 100 million identities). It is much faster than the model parallel solution and there is no performance drop.

![Image text](https://github.com/nttstar/insightface-resources/blob/master/images/partial_speed1.png)


## Contents
[Partial FC](https://arxiv.org/abs/2010.05222)
- [Largest Face Recognition Dataset: **Glint360k**](#Glint360k)
- [Installtion](./docs/installtion.md)
- [How to run](#Train)
- [Distributed Training Performance](#Performance)
- [Citation](#Citation)


## Glint360K
We clean, merge, and release the **largest** and **cleanest** face recognition dataset **Glint360K**, which contains 18 million images of 360K individuals. 
By employing the Patial FC training strategy, baseline models trained on Glint360K can easily achieve state-of-the-art performance. 
Detailed evaluation results on the large-scale test set (e.g. IFRT, IJB-C and Megaface) are as follows:

#### Evaluation on IFRT       
**`r`** denotes the sampling rate of negative class centers.
| Backbone     | Dataset            | African | Caucasian | Indian | Asian | ALL   |
| ------------ | -----------        | ----- | ----- | ------ | ----- | ----- |
| R50          | MS1M-V3            | 76.24 | 86.21 | 84.44  | 37.43 | 71.02 |
| R124         | MS1M-V3            | 81.08 | 89.06 | 87.53  | 38.40 | 74.76 |
| R100         | **Glint360k**(r=1.0)   | 89.50 | 94.23 | 93.54  | **65.07** | **88.67** |
| R100         | **Glint360k**(r=0.1)   | **90.45** | **94.60** | **93.96**  | 63.91 | 88.23 |

#### Evaluation on IJB-C and Megaface  
We employ ResNet100 as the backbone and CosFace (m=0.4) as the loss function.
TAR@FAR=1e-4 is reported on the IJB-C datasets, and TAR@FAR=1e-6 is reported on the Megaface dataset.
|Test Dataset        | IJB-C     | Megaface_Id  | Megaface_Ver |
| :---               | :---:     | :---:        | :---:        |
| MS1MV2             | 96.4      | 98.3         | 98.6         |
|**Glint360k** | **97.3**  | **99.1**     | **99.1**     |

#### Download
[**Baidu Drive**](https://pan.baidu.com/s/1aHC_nJGKzKgwJKoVb2Q_Gg) (code:i1al)  

Refer to the following command to unzip.
```
cat glint360k* > glint360k.tar
tar -xvf glint360k.tar
# md5sum:
# train.rec 2a74c71c4d20e770273f103eda97e878
# train.idx f7a3e98d3533ac481bdf3dc03a5416e8
```
Use [unpack_glint360k.py](./unpack_glint360k.py) to unpack.

## Train
#### Requirements
python==3.6  
cuda==10.1    
cudnn==765    
mxnet-cu101==1.6.0.post0  
pip install easydict mxboard opencv-python tqdm    
[nccl](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)  
[openmpi](./setup-utils/install-mpi.sh)==4.0.0  
[horovod](./setup-utils/install-horovod.sh)==0.19.2  

#### Failures due to SSH issues
The host where horovodrun is executed must be able to SSH to all other hosts without any prompts.

#### Run with horovodrun
Typically one GPU will be allocated per process, so if a server has 8 GPUs, you will run 8 processes. 
In horovodrun, the number of processes is specified with the -np flag.

To run on a machine with 8 GPUs:
```shell script
horovodrun -np 8 -H localhost:8 bash config.sh
```

To run on two machine with 16 GPUs:
```shell script
horovodrun -np 16 -H ip1:8,ip2:8 bash config.sh
```

#### Run with mpi
```shell script
bash run.sh
```

## Performance
We neglect the influence of IO. All experiments use mixed-precision training, and the backbone is ResNet50.
#### 1 Million Identities On 8 RTX2080Ti  

|Method                     | GPUs        | BatchSize     | Memory/M      | Throughput img/sec | W     |
| :---                      | :---:       | :---:         | :---:         | :---:              | :---: |
| Model Parallel            | 8           | 1024          | 10408         | 2390               | GPU   |
| **Partial FC(Ours)**      | **8**       | **1024**      | **8100**      | **2780**           | GPU   |
#### 10 Million Identities On 64 RTX2080Ti  

|Method                     | GPUs        | BatchSize     | Memory/M      | Throughput img/sec | W     |
| :---                      | :---:       | :---:         | :---:         | :---:              | :---: |
| Model Parallel            | 64          | 2048          | 9684          | 4483               | GPU   |
| **Partial FC(Ours)**      | **64**      | **4096**      | **6722**      | **12600**          | GPU   |



## Citation
If you find Partial-FC or Glint360K useful in your research, please consider to cite the following related paper: 

[Partial FC](https://arxiv.org/abs/2010.05222)
```
@inproceedings{an2020partical_fc,
  title={Partial FC: Training 10 Million Identities on a Single Machine},
  author={An, Xiang and Zhu, Xuhan and Xiao, Yang and Wu, Lan and Zhang, Ming and Gao, Yuan and Qin, Bin and
  Zhang, Debing and Fu Ying},
  booktitle={Arxiv 2010.05222},
  year={2020}
}
```




