# InsightFace 在 OneFlow 中的实现

本文将介绍如何在 OneFlow 中训练 InsightFace，并在验证数据集上对训练好的网络进行验证。

## 背景介绍

###  InsightFace 开源项目

[InsightFace 原仓库](https://github.com/deepinsight/insightface)是基于 MXNet 实现的人脸识别开源项目。

在该项目中，集成了：

* CASIA-Webface、MS1M、VGG2 等用于人脸识别研究常用的数据集（以 MXNet 支持的二进制的形式提供，从[这里](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)查看数据集的详细说明以及下载链接）。

* 以 ResNet、MobilefaceNet、InceptionResNet_v2 等深度学习网络作为 Backbone 的人脸识别模型。
* 涵盖 SphereFace Loss、Softmax Loss、SphereFace Loss 等多种损失函数的实现。



### InsightFace 在 OneFlow 中的实现

在 InsightFace 开源项目已有的工作基础上，OneFlow 对 InsightFace 基本的人脸识别模型进行了移植，目前已实现的功能包括：

* 支持了使用 MS1M、Glint360k 作为训练数据集，Lfw、Cfp_fp 以及 Agedb_30 作为验证数据集，提供了对网络进行训练和验证的脚本。
* 支持 ResNet100 和 MobileFaceNet 作为人脸识别模型的 Backbone 网络。
* 实现了 Softmax Loss 以及 Margin Softmax Loss（包括 Arcface、Cosface 和 Combined Loss）。
* 实现了模型并行和 Partial FC 优化。
* 实现了 MXNet 和 Onnx 的模型转换。


未来将计划逐步完善：

* 更多的数据集转换。
* 更丰富的 Backbone 网络。
* 更全面的损失函数实现。
* 增加分布式运行的说明。



我们对所有的开发者开放 PR，非常欢迎您加入新的实现以及参与讨论。

## 准备工作

在开始运行前，请先确定：

1. 安装 OneFlow。
2. 准备训练和验证的 OFRecord 数据集。



###  安装OneFlow

根据 [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) 的步骤进行安装最新 master whl 包即可。
```
python3 -m pip install --find-links https://release.oneflow.info oneflow_cu102 --user
```

### 准备数据集

根据 [加载与准备 OFRecord 数据集](https://docs.oneflow.org/extended_topics/how_to_make_ofdataset.html) 准备 ImageNet 的 OFReocord 数据集，用以进行 InsightFace 的测试。

[InsightFace 原仓库](https://github.com/deepinsight/insightface)中提供了一系列人脸识别任务相关的数据集，并且已经完成了人脸对齐等预处理过程。请从[这里](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)下载相应的数据集，并且转换成 OneFlow 可以识别的 OFRecord 格式。考虑到步骤繁琐，也可以直接下载已经转好的 OFRecord 数据集:[训练集](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/train_ofrecord.tar.gz)和[验证集](http://oneflow-public.oss-cn-beijing.aliyuncs.com/face_dataset/eval_ofrecord.tar.gz)。

下面分别数据集 MS1M-ArcFace 为例，展示如何将下载到的数据集转换成 OFRecord 格式。

#### 1. 下载数据集

下载好的 MS1M-ArcFace 数据集，目录如下：

```
faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```



前三个文件是训练数据集 MS1M 的 MXNet 的 recordio 格式相关的文件，后三个 `.bin` 文件是三个不同的验证数据集。



#### 2. 将训练数据集 MS1M 从 recordio 格式转换为 OFRecord 格式

运行：

```
python tools/dataset_convert/mx_recordio_2_ofrecord.py --data_dir datasets/faces_emore --output_filepath faces_emore/ofrecord/train
```



#### 3. 将验证数据集转换为 OFRecord 格式

运行：

```
python bin_2_ofrecord.py --data_dir=datasets/faces_emore --output_filepath=faces_emore/ofrecord/lfw/ --dataset_name="lfw"
python bin_2_ofrecord.py --data_dir=faces_emore --output_filepath=faces_emore/ofrecord/cfp_fp/ --dataset_name="cfp_fp"
python bin_2_ofrecord.py --data_dir=datasets/faces_emore --output_filepath=faces_emore/ofrecord/agedb_30/ --dataset_name="agedb_30"
```



## 训练和验证

### 训练

为了减小用户使用的迁移成本，OneFlow 的脚本已经修改为 MXNet 实现的风格，用户可以使用 sample_config.py 直接修改参数。同时，还可以通过添加命令行参数 `--do_validataion_while_train`，实现一边训练一边验证。

对于想要修改的参数可以直接在 sample_config.py 中修改。修改并拷贝 config.py 文件

```
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```

运行脚本：

```
python insightface_train.py --dataset emore  --network r100 --loss arcface
```

即可进行基于 face_emore 数据集使用 ResNet100 作为 Backbone 的训练和验证。

若想进行更大数据集的训练验证，运行脚本

```
python insightface_train.py --dataset glint360k_8GPU --network r100_glint360k --loss cosface 
```

即可进行基于 glint360k 数据集使用 ResNet100 作为 Backbone 的训练和验证。

### 单独执行验证

在上面训练的过程中，
另外，为了方便查看保存下来的预训练模型的精度，我们提供了一个仅在验证数据集上单独执行验证过程的脚本，入口文件为：`insightface_val.py `，使用方式如下：

```
python insightface_val.py \
--gpu_num_per_node=1 \
--network="r100" \
--model_load_dir=path/to/model_load_dir
```

其中，用 `--model_load_dir` 指定想要加载的预训练模型的路径。

精度结果记录可参考：https://github.com/Oneflow-Inc/oneflow_face/tree/partial_fc
