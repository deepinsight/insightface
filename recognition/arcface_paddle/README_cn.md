简体中文 | [English](README_en.md)

# Arcface-Paddle

* [1. 简介](#简介)
* [2. 环境准备](#环境准备)
* [3. 数据准备](#数据准备)
  * [3.1 下载数据集](#下载数据集)
  * [3.2 从 MXNet 格式数据集抽取图像](#从MXNet格式数据集抽取图像)
* [4. 训练](#训练)
  * [4.1 单机单卡](#单机单卡)
  * [4.2 单机 8 卡](#单机8卡)
* [5. 模型评价](#模型评价)
* [6. 模型导出](#模型导出)
* [7 模型推理](#模型推理)
* [8 模型性能](#模型性能)
  * [8.1 轻量化模型性能](#轻量化模型性能)
  * [8.2 验证集准确率](#验证集准确率)
  * [8.3 最大类别数支持](#最大类别数支持)
  * [8.4 吞吐量对比](#吞吐量对比)
* [9. 全流程推理](#全流程推理)

<a name="简介"></a>

## 1. 简介

`Arcface-Paddle`是基于PaddlePaddle实现的，开源深度人脸检测、识别工具。`Arcface-Paddle`目前提供了三个预训练模型，包括用于人脸检测的 `BlazeFace`、用于人脸识别的 `ArcFace` 和 `MobileFace`。

- 本部分内容为人脸识别部分。
- 人脸检测相关内容可以参考：[基于BlazeFace的人脸检测](../../detection/blazeface_paddle/README_cn.md)。
- 基于PaddleInference的Whl包预测部署内容可以参考：[Whl包预测部署](https://github.com/littletomatodonkey/insight-face-paddle)。

注: 在此非常感谢 [GuoQuanhao](https://github.com/GuoQuanhao) 基于PaddlePaddle复现了 [Arcface的基线模型](https://github.com/GuoQuanhao/arcface-Paddle)。

<a name="环境准备"></a>

## 2. 环境准备

请参照 [Installation](./install_cn.md) 配置实验所需环境。

<a name="数据准备"></a>

## 3. 数据准备

<a name="下载数据集"></a>

### 3.1 下载数据集

数据集可以从 [insightface datasets](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) 下载.

* MS1M_v2: MS1M-ArcFace
* MS1M_v3: MS1M-RetinaFace

<a name="从MXNet格式数据集抽取图像"></a>

### 3.2 从 MXNet 格式数据集抽取图像

```shell
python tools/mx_recordio_2_images.py --root_dir ms1m-retinaface-t1/ --output_dir MS1M_v3/
```

当数据集抽取完后，输出的图像数据集目录结构如下：

```
MS1M_v3
|_ images
|  |_ 00000001.jpg
|  |_ ...
|  |_ 05179510.jpg
|_ label.txt
|_ agedb_30.bin
|_ cfp_ff.bin
|_ cfp_fp.bin
|_ lfw.bin
```

标签文件格式如下：

```
# 图像路径与标签的分隔符: "\t"
# 以下是 label.txt 每行的格式
images/00000001.jpg 0
...
```

如果你想使用自定义数据集训练，可以根据以上目录结构和标签文件格式组织数据。

<a name="训练"></a>

## 4. 训练

<a name="单机单卡"></a>

### 4.1 单机单卡

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py \
    --config_file configs/ms1mv2_mobileface.py \
    --embedding_size 128 \
    --sample_ratio 1.0 \
    --loss ArcFace \
    --batch_size 512 \
    --dataset MS1M_v2 \
    --num_classes 85742 \
    --data_dir MS1M_v2/ \
    --label_file MS1M_v2/label.txt \
    --fp16 False
```
<a name="单机8卡"></a>

### 4.2 单机 8 卡

为了方便训练，已经为用户准备好训练启动脚本。

#### 静态图模式训练

```bash
sh scripts/train_static.sh
```

#### 动态图模式训练

```bash
sh scripts/train_dynamic.sh
```

注：多机器多卡训练参见 ``paddle.distributed.launch`` API 文档。单机与多机训练不同之处在于多机需要设置 ``--ips`` 参数。

在训练过程中，你可以实时通过 `VisualDL` 可视化查看 loss 的变化，更多信息可以参考 [VisualDL](https://github.com/PaddlePaddle/VisualDL/)。

<a name="模型评价"></a>

## 5. 模型评价

模型评价可以通过以下脚本启动

#### 静态图模式

```bash
sh scripts/validation_static.sh
```

#### 动态图模式

```bash
sh scripts/validation_dynamic.sh
```

<a name="模型导出"></a>

## 6. 模型导出

PaddlePaddle 支持用预测引擎直接推理，首先，需要导出推理模型，通过以下脚本进行导出

#### 静态图模式

```bash
sh scripts/export_static.sh
```

#### 动态图模式

```bash
sh scripts/export_dynamic.sh
```

<a name="模型推理"></a>

## 7. 模型推理

模型推理过程支持 paddle 格式的 ``save inference model`` 和 onnx 格式。

```bash
sh scripts/inference.sh
```

<a name="模型性能"></a>

## 8. 模型性能

<a name="轻量化模型性能"></a>

### 8.1 轻量化模型性能

**配置：**
  * CPU: Intel(R) Xeon(R) Gold 6184 CPU @ 2.40GHz
  * GPU: a single NVIDIA Tesla V100
  * Precison: FP32
  * BatchSize: 64/512
  * SampleRatio: 1.0
  * Embedding Size: 128
  * MS1MV2

| Model structure           | lfw    | cfp_fp  | agedb30 | CPU time cost | GPU time cost | Inference model |
| ------------------------- | ------ | ------- | ------- | -------| -------- |---- |
| MobileFace-Paddle      | 0.9952 | 0.9280  | 0.9612  | 4.3ms  | 2.3ms    | [download link](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/mobileface_v1.0_infer.tar)  |
| MobileFace-mxnet       | 0.9950 | 0.8894  | 0.9591  | 7.3ms  | 4.7ms    | -   |

* 注: MobileFace-Paddle 是使用 MobileFaceNet_128 backbone 训练出的模型

<a name="验证集准确率"></a>

### 8.2 验证集准确率

**配置：**
  * GPU: 8 NVIDIA Tesla V100 32G
  * Precison: Pure FP16
  * BatchSize: 128/1024

| Mode    | Datasets | backbone | Ratio | agedb30 | cfp_fp | lfw  | log  | checkpoint |
| ------- | :------: | :------- | ----- | :------ | :----- | :--- | :--- |  :--- |
| Static  |  MS1MV3  | r50      | 0.1   | 0.98317 | 0.98943| 0.99850 | [log](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/logs/static/ms1mv3_r50_static_128_fp16_0.1/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_static_128_fp16_0.1_epoch_24.tgz) |
| Static  |  MS1MV3  | r50      | 1.0   | 0.98283 | 0.98843| 0.99850 | [log](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/logs/static/ms1mv3_r50_static_128_fp16_1.0/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_static_128_fp16_1.0_epoch_24.tgz) |
| Dynamic |  MS1MV3  | r50      | 0.1   | 0.98333 | 0.98900| 0.99833 | [log](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/logs/dynamic/ms1mv3_r50_dynamic_128_fp16_0.1/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_dynamic_128_fp16_0.1_eopch_24.tgz) |
| Dynamic |  MS1MV3  | r50      | 1.0   | 0.98317 | 0.98900| 0.99833 | [log](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/logs/dynamic/ms1mv3_r50_dynamic_128_fp16_1.0/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_dynamic_128_fp16_1.0_eopch_24.tgz) |

<a name="最大类别数支持"></a>

### 8.3 最大类别数支持 

**配置：**
  * GPU: 8 NVIDIA Tesla V100 32G (32510MiB)
  * BatchSize: 64/512
  * SampleRatio: 0.1

| Mode                      | Precision | Res50    | Res100   |
| ------------------------- | --------- | -------- | -------- |
| Framework1 (static)       | AMP       | 42000000 (31792MiB)| 39000000 (31938MiB)|
| Framework2 (dynamic)      | AMP       | 30000000 (31702MiB)| 29000000 (32286MiB)|
| Paddle (static)           | Pure FP16 | 60000000 (32018MiB)| 60000000 (32018MiB)|
| Paddle (dynamic)          | Pure FP16 | 59000000 (31970MiB)| 59000000 (31970MiB)|

* 注：在跑实验前配置环境变量 ``export FLAGS_allocator_strategy=naive_best_fit``

<a name="吞吐量对比"></a>

### 8.4 吞吐量对比

**配置：**
  * BatchSize: 128/1024
  * SampleRatio: 0.1
  * Datasets: MS1MV3
  * V100: Driver Version: 450.80.02, CUDA Version: 11.0
  * A100: Driver Version: 460.32.03, CUDA Version: 11.2
  
![insightface_throughtput](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/images/insightface_throughtput.png)

更多实验结果可以参考 [PLSC](https://github.com/PaddlePaddle/PLSC)，PLSC (Paddle Large Scale Classification) 是 Paddle 官方开源的大规模分类库，支持单机 8 卡 NVIDIA V100 (32G) 训练 6000 千万类，目前还在持续更新中，请关注。

<a name="全流程推理"></a>

## 9. 全流程推理

首先下载索引库、待识别图像与字体文件。

```bash
# 下载用于人脸识别的索引库，这里因为示例图像是老友记中的图像，所以使用老友记中角色的人脸图像构建的底库。
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/index.bin
# 下载用于人脸识别的示例图像
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/query/friends2.jpg
# 下载字体，用于可视化
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/SourceHanSansCN-Medium.otf
```

`检测+识别` 串联预测的示例脚本如下。

```shell
# 同时使用检测+识别
python3.7 tools/test_recognition.py --det --rec --index=index.bin --input=friends2.jpg --output="./output"
```

最终可视化结果保存在`output`目录下，可视化结果如下所示。

<div align="center">
<img src="https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/output/friends2.jpg"  width = "800" />
</div>

更多关于参数解释，索引库构建、whl包预测部署和Paddle Serving预测部署的内容可以参考：
 * [Whl包预测部署](https://github.com/littletomatodonkey/insight-face-paddle)
 * [Paddle Serving预测部署](./deploy/pdserving/README_CN.md)
