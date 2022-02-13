## Partial-FC
Partial FC is a distributed deep learning training framework for face recognition. The goal of Partial FC is to facilitate large-scale classification task (e.g. 10 or 100 million identities). It is much faster than the model parallel solution and there is no performance drop.

![Image text](https://github.com/nttstar/insightface-resources/blob/master/images/partial_speed1.png)


## Contents
[Partial FC](https://arxiv.org/abs/2010.05222)
- [Largest Face Recognition Dataset: **Glint360k**](#Glint360K)
- [Docker](#Docker)
- [Performance On Million Identities](#Benchmark)
- [FAQ](#FAQ)
- [Citation](#Citation)


## Glint360K
We clean, merge, and release the largest and cleanest face recognition dataset Glint360K, 
which contains **`17091657`** images of **`360232`** individuals. 
By employing the Patial FC training strategy, baseline models trained on Glint360K can easily achieve state-of-the-art performance. 
Detailed evaluation results on the large-scale test set (e.g. IFRT, IJB-C and Megaface) are as follows:

### 1. Evaluation on IFRT       
**`r`** denotes the sampling rate of negative class centers.
| Backbone     | Dataset            | African | Caucasian | Indian | Asian | ALL   |
| ------------ | -----------        | ----- | ----- | ------ | ----- | ----- |
| R50          | MS1M-V3            | 76.24 | 86.21 | 84.44  | 37.43 | 71.02 |
| R124         | MS1M-V3            | 81.08 | 89.06 | 87.53  | 38.40 | 74.76 |
| R100         | **Glint360k**(r=1.0)   | 89.50 | 94.23 | 93.54  | **65.07** | **88.67** |
| R100         | **Glint360k**(r=0.1)   | **90.45** | **94.60** | **93.96**  | 63.91 | 88.23 |

### 2. Evaluation on IJB-C and Megaface  
We employ ResNet100 as the backbone and CosFace (m=0.4) as the loss function.
TAR@FAR=1e-4 is reported on the IJB-C datasets, and TAR@FAR=1e-6 is reported on the Megaface dataset.
|Test Dataset        | IJB-C     | Megaface_Id  | Megaface_Ver |
| :---               | :---:     | :---:        | :---:        |
| MS1MV2             | 96.4      | 98.3         | 98.6         |
|**Glint360k** | **97.3**  | **99.1**     | **99.1**     |

### 3. License 

The Glint360K dataset (and the models trained with this dataset) are available for non-commercial research purposes only.

### 4. Download
- [x] [**Baidu Drive**](https://pan.baidu.com/s/1GsYqTTt7_Dn8BfxxsLFN0w) (code:o3az)    
- [x] **Magnet URI**: `magnet:?xt=urn:btih:E5F46EE502B9E76DA8CC3A0E4F7C17E4000C7B1E&dn=glint360k`

Refer to the following command to unzip.
```
cat glint360k_* | tar -xzvf -

# Don't forget the last '-'!

# cf7433cbb915ac422230ba33176f4625  glint360k_00
# 589a5ea3ab59f283d2b5dd3242bc027a  glint360k_01
# 8d54fdd5b1e4cd55e1b9a714d76d1075  glint360k_02
# cd7f008579dbed9c5af4d1275915d95e  glint360k_03
# 64666b324911b47334cc824f5f836d4c  glint360k_04
# a318e4d32493dd5be6b94dd48f9943ac  glint360k_05
# c3ae1dcbecea360d2ec2a43a7b6f1d94  glint360k_06
# md5sum:
# 5d9cd9f262ec87a5ca2eac5e703f7cdf train.idx
# 8483be5af6f9906e19f85dee49132f8e train.rec
```
Use [unpack_glint360k.py](./unpack_glint360k.py) to unpack.

### 5. Pretrain models
- [x] [**Baidu Drive**](https://pan.baidu.com/s/1sd9ZRsV2c_dWHW84kz1P1Q) (code:befi)
- [x] [**Google Drive**](https://drive.google.com/drive/folders/1WLjDzEs1wC1K1jxDHNJ7dhEmQ3rOOILl?usp=sharing)

| Framework       |  backbone                      | negative class centers sample_rate  | IJBC@e4 | IFRT@e6 |
|  :---           | :---                           | :---         |  :---   |  :---   | 
|  mxnet   | [R100](https://drive.google.com/drive/folders/1YPqIkOZWrmbli4GWfMJO2b0yiiZ7UCsP?usp=sharing) |1.0|97.3|-|
|  mxnet   | [R100](https://drive.google.com/drive/folders/1-gF5sDwNoRcjwmpPSTNLpaZJi5N91BvL?usp=sharing) |0.1|97.3|-|
|  pytorch | [R50](https://drive.google.com/drive/folders/16hjOGRJpwsJCRjIBbO13z3SrSgvPTaMV?usp=sharing) |1.0|97.0|-|    
|  pytorch | [R100](https://drive.google.com/drive/folders/19EHffHN0Yn8DjYm5ofrgVOf_xfkrVgqc?usp=sharing) |1.0|97.4|-|
    
## Docker
Make sure you have installed the NVIDIA driver and Docker engine for your Linux distribution Note that you do not need to 
install the CUDA Toolkit and other independence on the host system, but the NVIDIA driver needs to be installed.  
Because the CUDA version used in the image is 10.1, 
the graphics driver version on the physical machine must be greater than 418.

### 1. Docker Getting Started
You can use dockerhub or offline docker.tar to get the image of the Partial-fc.
1. dockerhub
```shell
docker pull insightface/partial_fc:v1
```  

2. offline images  
coming soon!

### 2. Getting Started
```shell
sudo docker run -it -v /train_tmp:/train_tmp --net=host --privileged --gpus 8 --shm-size=1g insightface/partial_fc:v1 /bin/bash
```

`/train_tmp` is where you put your training set (if you have enough RAM memory, 
you can turn it into `tmpfs` first).

## Benchmark
### 1. Train Glint360K Using MXNET
 
| Backbone    |   GPU                       | FP16  | BatchSize / it | Throughput img / sec |
|  :---       | :---                        | :---  |   :---         | :---                 | 
|  R100       | 8 * Tesla V100-SXM2-32GB    | False |   64           | 1748                 |
|  R100       | 8 * Tesla V100-SXM2-32GB    | True  |   64           | 3357                 |
|  R100       | 8 * Tesla V100-SXM2-32GB    | False |   128          | 1847                 |    
|  R100       | 8 * Tesla V100-SXM2-32GB    | True  |   128          | 3867                 |   
|  R50        | 8 * Tesla V100-SXM2-32GB    | False |   64           | 2921                 |
|  R50        | 8 * Tesla V100-SXM2-32GB    | True  |   64           | 5428                 |
|  R50        | 8 * Tesla V100-SXM2-32GB    | False |   128          | 3045                 |    
|  R50        | 8 * Tesla V100-SXM2-32GB    | True  |   128          | 6112                 |  


### 2. Performance On Million Identities
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


## FAQ
#### Glint360K's Face Alignment Settings?
We use a same alignment setting with MS1MV2, code is [here](https://github.com/deepinsight/insightface/issues/1286).

#### Why update Glint360K, is there a bug in the previous version?  
In the previous version of Glint360K, there is no bug when using softmax training, but there is a bug in triplet training. 
In the latest Glint360k, this bug has been fixed.

#### Dataset in Google Drive or Dropbox?
The torrent has been released.


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




