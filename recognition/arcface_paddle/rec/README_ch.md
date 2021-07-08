简体中文 | [English](README_en.md)

# Arcface-Paddle

首先，请参照 [Installation](../install_ch.md) 配置实验所需环境。

## 1. 数据准备

### 1.1 进入 repo 目录。

```
cd arcface_paddle/rec
```

### 1.2 下载与解压数据集

使用下面的命令下载并解压 MS1M 数据集。

```shell
# 下载数据集
wget https://paddle-model-ecology.bj.bcebos.com/data/insight-face/MS1M_bin.tar
# 解压数据集
tar -xf MS1M_bin.tar
```

注意：
* 如果希望在windows环境下安装wget，可以参考：[链接](https://www.cnblogs.com/jeshy/p/10518062.html)；如果希望在windows环境中安装tar命令，可以参考：[链接](https://www.cnblogs.com/chooperman/p/14190107.html)。
* 如果macOS环境下没有安装wget命令，可以运行下面的命令进行安装。

```shell
# 安装 homebrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)";
# 安装wget
brew install wget
```


解压完成之后，文件夹目录结构如下。

```
Arcface-Paddle/MSiM_bin
|_ images
|  |_ 00000000.bin
|  |_ ...
|  |_ 05822652.bin
|_ label.txt
|_ agedb_30.bin
|_ cfp_ff.bin
|_ cfp_fp.bin
|_ lfw.bin
```

* 标签文件格式：

  ```
   # delimiter: "\t"
   # the following the content of label.txt
   images/00000000.bin 0
   ...
  ```

如果需要使用自定义数据集，请按照上述格式进行整理，并替换配置文件中的数据集目录。

注意：
* 这里为了更加方便`Dataloader`读取数据，将原始的`train.rec`文件转化为很多`bin文件`，每个`bin文件`都唯一对应一张原始图像。如果您采集得到的文件均为原始的图像文件，那么可以参考`1.3节`中的内容完成原始图像文件到bin文件的转换。

### 1.3 原始图像文件与bin文件的转换

如果希望将原始的图像文件转换为本文用于训练的bin文件，那么可以使用下面的命令进行转换。

```shell
python3.7 tools/convert_image_bin.py --image_path="your/input/image/path" --bin_path="your/output/bin/path" --mode="image2bin"
```

如果希望将bin文件转化为原始的图像文件，那么可以使用下面的命令进行转换。

```shell
python3.7 tools/convert_image_bin.py --image_path="your/input/bin/path" --bin_path="your/output/image/path" --mode="bin2image"
```

## 2. 模型训练

准备好配置文件后，可以通过以下方式开始训练过程。

```bash
python3.7 train.py \
    --network 'MobileFaceNet_128' \
    --lr=0.1 \
    --batch_size 512 \
    --weight_decay 2e-4 \
    --embedding_size 128 \
    --logdir="log" \
    --output "emore_arcface" \
    --resume 0
```

上述命令中，需要传入如下参数:

+ `network`: 模型名称, 默认值为 `MobileFaceNet_128`;
+ `lr`: 初始学习率, 默认值为  `0.1`;
+ `batch_size`:  Batch size 的大小, 默认值为  `512`;
+ `weight_decay`:  正则化策略, 默认值为  `2e-4`;
+ `embedding_size`: 人脸 embedding 的长度, 默认值为 `128`;
+ `logdir`: VDL 输出 log 的存储路径, 默认值为 `"log"`;
+ `output`: 训练过程中的模型文件存储路径, 默认值为 `"emore_arcface"`;
+ `resume`: 是否恢复分类层的模型权重。 `1` 表示使用之前好的权重文件进行初始化，  `0` 代表重新初始化。 如果想要恢复分类层的模型权重， 需要保证 `output` 目录下包含： `rank:0_softmax_weight_mom.pkl` 和 `rank:0_softmax_weight.pkl` 两个文件。

* 训练过程中的输出 log 示例如下:

  ```
  ...
  Speed 500.89 samples/sec   Loss 55.5692   Epoch: 0   Global Step: 200   Required: 104 hours, lr_backbone_value: 0.000000, lr_pfc_value: 0.000000
  ...
  [lfw][2000]XNorm: 9.890562
  [lfw][2000]Accuracy-Flip: 0.59017+-0.02031
  [lfw][2000]Accuracy-Highest: 0.59017
  [cfp_fp][2000]XNorm: 12.920007
  [cfp_fp][2000]Accuracy-Flip: 0.53329+-0.01262
  [cfp_fp][2000]Accuracy-Highest: 0.53329
  [agedb_30][2000]XNorm: 12.188049
  [agedb_30][2000]Accuracy-Flip: 0.51967+-0.02316
  [agedb_30][2000]Accuracy-Highest: 0.51967
  ...
  ```


在训练过程中，可以通过  `VisualDL` 实时查看loss变化，更多信息请参考 [VisualDL](https://github.com/PaddlePaddle/VisualDL/)。


## 3. 模型评估

可以通过以下方式开始模型评估过程。

```bash
python3.7 valid.py
    --network MobileFaceNet_128  \
    --checkpoint emore_arcface \
```

上述命令中，需要传入如下参数:

+ `network`: 模型名称, 默认值为 `MobileFaceNet_128`;
+ `checkpoint`: 保存模型权重的目录, 默认值为 `emore_arcface`;

**注意:** 上面的命令将评估模型文件 `./emore_arcface/MobileFaceNet_128.pdparams` .您也可以通过同时修改 `network` 和 `checkpoint` 来修改要评估的模型文件。

## 4. 模型精度与速度评估

在MS1M训练集上进行模型训练，最终得到的模型指标在lfw、cfp_fp、agedb30三个数据集上的精度指标以及CPU、GPU的预测耗时如下。

| 模型结构                  | lfw   | cfp_fp | agedb30  | CPU 耗时 | GPU 耗时 |
| ------------------------- | ----- | ------ | ------- |-------|  -------- |
| MobileFaceNet-Paddle      | 0.9945 | 0.9343  | 0.9613 | 4.3ms | 2.3ms   |
| MobileFaceNet-mxnet | 0.9950 | 0.8894  | 0.9591   |  7.3ms | 4.7ms   |

**测试环境：**
  * CPU: Intel(R) Xeon(R) Gold 6184 CPU @ 2.40GHz
  * GPU: a single NVIDIA Tesla V100


## 5. 模型导出
PaddlePaddle支持使用预测引擎进行预测推理，通过导出inference模型将模型固化：

```bash
python export_inference_model.py --network MobileFaceNet_128 --output ./inference_model/ --pretrained_model ./emore_arcface/MobileFaceNet_128.pdparams
```

导出模型后，在 `./inference_model/` 目录下有：

```
./inference_model/
|_ inference.pdmodel
|_ inference.pdiparams
```
