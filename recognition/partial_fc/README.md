## Partial-FC
Partial FC is a distributed deep learning training framework for face recognition. The goal of Partial FC is to facilitate large-scale classification task (e.g. 10 or 100 million identities). It is much faster than the model parallel solution and there is no performance drop.

![Image text](https://github.com/nttstar/insightface-resources/blob/master/images/partial_speed1.png)


## Contents
[Partial FC](https://arxiv.org/abs/2010.05222)
- [Largest Face Recognition Dataset: **Glint360k**](#Glint360k)
- [Distributed Training Performance](#Performance)
- [FAQ](#FAQ)
- [Citation](#Citation)


## Glint360K
We clean, merge, and release the largest and cleanest face recognition dataset Glint360K, 
which contains **`17091657`** images of **`360232`** individuals. 
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
[**Baidu Drive**](https://pan.baidu.com/s/1GsYqTTt7_Dn8BfxxsLFN0w) (code:o3az)    
**Magnet URI:** magnet:?xt=urn:btih:E5F46EE502B9E76DA8CC3A0E4F7C17E4000C7B1E&dn=glint360k  


Refer to the following command to unzip.
```
cf7433cbb915ac422230ba33176f4625  glint360k_00
589a5ea3ab59f283d2b5dd3242bc027a  glint360k_01
8d54fdd5b1e4cd55e1b9a714d76d1075  glint360k_02
cd7f008579dbed9c5af4d1275915d95e  glint360k_03
64666b324911b47334cc824f5f836d4c  glint360k_04
a318e4d32493dd5be6b94dd48f9943ac  glint360k_05
c3ae1dcbecea360d2ec2a43a7b6f1d94  glint360k_06

cat glint360k_* | tar -xzvf -
# md5sum:
5d9cd9f262ec87a5ca2eac5e703f7cdf train.idx
8483be5af6f9906e19f85dee49132f8e train.rec
```
Use [unpack_glint360k.py](./unpack_glint360k.py) to unpack.

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

## FAQ
#### Glint360K's Face Alignment Settings?
We use a same alignment setting with MS1MV2, code is [here](https://github.com/deepinsight/insightface/issues/1286).

#### Why update Glint360K, is there a bug in the previous version?  
In the previous version of Glint360K, there is no bug when using softmax training, but there is a bug in triplet training. 
In the latest Glint360k, this bug has been fixed.

#### Dataset in Google Drive or Dropbox?
The torrent will be release soon.


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




