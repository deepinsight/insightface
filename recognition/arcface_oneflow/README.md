
# InsightFace in OneFlow

[English](README.md) **|** [简体中文](README_CH.md)

It introduces how to train InsightFace in OneFlow, and do verification over the validation datasets via the well-toned networks.

## Contents

\- [InsightFace in OneFlow](#insightface-in-oneflow)

 \- [Contents](#contents)

 \- [Background](#background)

  \- [InsightFace opensource project](#insightface-opensource-project)

  \- [Implementation in OneFlow](#implementation-in-oneflow)

 \- [Preparations](#preparations)

  \- [Install OneFlow](#install-oneflow)

  \- [Data preparations](#data-preparations)

   \- [1. Download datasets](#1-download-datasets)

   \- [2. Transformation from MS1M recordio to OFRecord](#2-transformation-from-ms1m-recordio-to-ofrecord)

 \- [Training and verification](#training-and-verification)

  \- [Training](#training)

  \- [OneFLow2ONNX](#OneFLow2ONNX)

## Background

### InsightFace opensource project

[InsightFace](https://github.com/deepinsight/insightface) is an open-source 2D&3D deep face analysis toolbox, mainly based on MXNet.

In InsightFace, it supports:



- Datasets typically used for face recognition, such as CASIA-Webface、MS1M、VGG2(Provided with the form of a binary file which could run in MXNet, [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) is more details about the datasets and how to download.



* Backbones of ResNet, MobilefaceNet, InceptionResNet_v2, and other deep-learning networks to apply in facial recognition. 

* Implementation of different loss functions, including SphereFace Loss、Softmax Loss、SphereFace Loss, etc.

  

### Implementation in OneFlow

Based upon the currently existing work of Insightface, OneFlow ported basic models from it, and now OneFlow supports:



- Training datasets of MS1M、Glint360k, and validation datasets of Lfw、Cfp_fp and Agedb_30, scripts for training and validating.

- Backbones of ResNet100 and MobileFaceNet to recognize faces.

- Loss function, e.g. Softmax Loss and Margin Softmax Loss（including Arcface、Cosface and Combined Loss）.

- Model parallelism and [Partial FC](https://github.com/deepinsight/insightface/tree/760d6de043d7f654c5963391271f215dab461547/recognition/partial_fc#partial-fc) optimization.

- Model transformation via MXNet.



To be coming further:

- Additional datasets transformation.

- Plentiful backbones.

- Full-scale loss functions implementation.

- Incremental tutorial on the distributed configuration.



This project is open for every developer to PR, new implementation and animated discussion will be most welcome.



## Preparations

First of all, before execution, please make sure that:

1. Install OneFlow

2. Prepare training and validation datasets in form of OFRecord.



### Install OneFlow



According to steps in [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) install the newest release master whl packages.

```
python3 -m pip install oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu102/6aa719d70119b65837b25cc5f186eb19ef2b7891/index.html --user
```



### Data preparations

According to [Load and Prepare OFRecord Datasets](https://docs.oneflow.org/en/extended_topics/how_to_make_ofdataset.html), datasets should be converted into the form of OFREcord, to test InsightFace.



It has provided a set of datasets related to face recognition tasks, which have been pre-processed via face alignment or other processions already in [InsightFace](https://github.com/deepinsight/insightface). The corresponding datasets could be downloaded from [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) and should be converted into OFRecord, which performs better in OneFlow. Considering the cumbersome steps, it is suggested to download converted OFrecord datasets:

[MS1M-ArcFace(face_emore)](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/train_ofrecord.tar.gz)

[MS1MV3](https://oneflow-public.oss-cn-beijing.aliyuncs.com/facedata/MS1V3/oneflow/ms1m-retinaface-t1.zip)

It illustrates how to convert downloaded datasets into OFRecords, and take MS1M-ArcFace as an example in the following.

#### 1. Download datasets

The structure of the downloaded MS1M-ArcFace is shown as follown：



```
faces_emore/

​    train.idx

​    train.rec

​    property

​    lfw.bin

​    cfp_fp.bin

​    agedb_30.bin
```

The first three files are MXNet recordio format files of MS1M training dataset, the last three `.bin` files are different validation datasets.



#### 2. Transformation from MS1M recordio to OFRecord
Only need to execute 2.1 or 2.2
2.1 Use Python scripts directly

Run 
```
python tools/mx_recordio_2_ofrecord_shuffled_npart.py  --data_dir datasets/faces_emore --output_filepath faces_emore/ofrecord/train --num_part 16
```
And you will get the number of `part_num` parts of OFRecord, it's 16 parts in this example, it showed like this
```
tree ofrecord/test/
ofrecord/test/
|-- _SUCCESS
|-- part-00000
|-- part-00001
|-- part-00002
|-- part-00003
|-- part-00004
|-- part-00005
|-- part-00006
|-- part-00007
|-- part-00008
|-- part-00009
|-- part-00010
|-- part-00011
|-- part-00012
|-- part-00013
|-- part-00014
`-- part-00015

0 directories, 17 files
```


2.2 Use Python scripts + Spark Shuffle + Spark partition

Run

```
python tools/dataset_convert/mx_recordio_2_ofrecord.py --data_dir datasets/faces_emore --output_filepath faces_emore/ofrecord/train
```

And you will get one part of OFRecord(`part-0`) with all data in this way. Then you should use Spark to shuffle and partition.
1. Get jar package available
You can download Spark-oneflow-connector-assembly-0.1.0.jar via [Github](https://github.com/Oneflow-Inc/spark-oneflow-connector) or [OSS](https://oneflow-public.oss-cn-beijing.aliyuncs.com/spark-oneflow-connector/spark-oneflow-connector-assembly-0.1.1.jar)

2. Run in Spark
Assign that you have already installed and configured Spark.
Run
```
//Start Spark 
./Spark-2.4.3-bin-hadoop2.7/bin/Spark-shell --jars ~/Spark-oneflow-connector-assembly-0.1.0.jar --driver-memory=64G --conf Spark.local.dir=/tmp/
// shuffle and partition in 16 parts
import org.oneflow.Spark.functions._
Spark.read.chunk("data_path").shuffle().repartition(16).write.chunk("new_data_path")
sc.formatFilenameAsOneflowStyle("new_data_path")
```
Hence you will get 16 parts of OFRecords, it shown like this
```
tree ofrecord/test/
ofrecord/test/
|-- _SUCCESS
|-- part-00000
|-- part-00001
|-- part-00002
|-- part-00003
|-- part-00004
|-- part-00005
|-- part-00006
|-- part-00007
|-- part-00008
|-- part-00009
|-- part-00010
|-- part-00011
|-- part-00012
|-- part-00013
|-- part-00014
`-- part-00015

0 directories, 17 files
```


## Training and verification



### Training

To reduce the usage cost of user, OneFlow draws close the scripts to Torch style, you can directly modify parameters via configs/*.py

#### eager 
```
./train_ddp.sh
```
#### Graph
```
train_graph_distributed.sh
```


### Varification

Moreover, OneFlow offers a validation script to do verification separately, val.py, which facilitates you to check the precision of the pre-training model saved.

```
./val.sh

```
## OneFLow2ONNX

```
pip install oneflow-onnx==0.5.1
./convert.sh
```