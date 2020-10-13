## Partial-FC
Partial FC is a distributed deep learning training framework for face recognition. The goal of Partial FC is to make 
training large scale classification task (eg. 10 or 100 millions identies) fast and easy. It is faster than model parallel 
and can train more identities.  

![Image text](https://github.com/nttstar/insightface-resources/blob/master/images/partial_speed1.png)


## Contents
[Partial FC](#Partial)
- [Largest Face Recognition Dataset: **Glint360k**](#Glint360k)
- [Installtion](./docs/installtion.md)
- [How to run](./docs/how_to_run.md)
- [Distributed Training Performance](#Performance)
- [Citation](#Citation)


## Glint360k
We clean, merge, and release the **largest** and **cleanest** face recognition dataset **Glint360k**. 
Baseline models trained on Glint360k with our proposed training strategy can easily achieve state-of-the-art. 
The released dataset contains 18 million images of 360K individuals. The performance of Glint360k eval on large-scale 
test set IFRT, IJB-C and Megaface are as follows:

#### Evaluation on IFRT       
**`r`** means the negative class centers sampling rate.
| Backbone     | Dataset            | African | Caucasian | Indian | Asian | ALL   |
| ------------ | -----------        | ----- | ----- | ------ | ----- | ----- |
| R50          | MS1M-V3            | 76.24 | 86.21 | 84.44  | 37.43 | 71.02 |
| R124         | MS1M-V3            | 81.08 | 89.06 | 87.53  | 38.40 | 74.76 |
| R100         | **Glint360k**(r=1.0)   | 89.50 | 94.23 | 93.54  | **65.07** | **88.67** |
| R100         | **Glint360k**(r=0.1)   | **90.45** | **94.60** | **93.96**  | 63.91 | 88.23 |

#### Evaluation on IJB-C and Megaface  

Our backbone is ResNet100, we set feature scale s to 64 and cosine margin m of CosFace at 0.4.
TAR@FAR=1e-4 is reported on the IJB-C datasets, TAR@FAR=1e-6 is reported on Megaface verification.
|Test Dataset        | IJB-C     | Megaface_Id  | Megaface_Ver |
| :---               | :---:     | :---:        | :---:        |
| MS1MV2             | 96.4      | 98.3         | 98.6         |
|**Glint360k** | **97.3**  | **99.1**     | **99.1**     |

#### Download
[**Baidu Pan**](https://pan.baidu.com/s/1aHC_nJGKzKgwJKoVb2Q_Gg) (code:i1al)  
Google Drive  coming soon

Refer to the following command to unzip.
```
cat glint360k* > glint360k.tar
tar -xvf glint360k.tar
# md5sum:
# train.rec 2a74c71c4d20e770273f103eda97e878
# train.idx f7a3e98d3533ac481bdf3dc03a5416e8
```


## Performance
We remove the influence of IO, all experiments use mixed precision training, backbone is ResNet50.
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
If you find Partical-FC or Glint360k useful in your research, please consider to cite the following related papers:
```
@inproceedings{an2020partical_fc,
  title={Partial FC: Training 10 Million Identities on a Single Machine},
  author={Xiang An},
  booktitle={Arxiv},
  year={2020}
}
```




