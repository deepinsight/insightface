# 飞桨训推一体认证

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了 ArcFace 中所有 PaddlePaddle 模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

<div align="center">
    <img src="docs/guide.png" width="1000">
</div>

## 2. 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：包括模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度。
- 模型压缩：包括裁剪、离线/在线量化、蒸馏。
- 其他预测部署：包括Paddle Inference C++预测、Paddle Serving部署、Paddle-Lite部署等。

更详细的mkldnn、Tensorrt等预测加速相关功能的支持情况可以查看各测试工具的[更多教程](#more)。

| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 模型压缩 |  其他预测部署  |
| :---:| :---: |  :----:  | :--------: |  :----:  |   :----:  |   :----:  |
| ArcFace     | ms1mv2_mobileface | 识别  | 支持 | 多机多卡 | - | Paddle Serving: Python |



## 3. 一键测试工具使用
### 目录介绍

```shell
test_tipc/
├── configs/  # 配置文件目录
	├── ms1mv2_mobileface  # ms1mv2_mobileface 模型的测试配置文件目录
		├── model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt # 测试Linux上python serving预测的配置文件
		└── train_infer_python.txt # 测试Linux上python训练预测（基础训练预测）的配置文件
	├── ...  
├── data/ # 存放 TIPC 测试数据的目录
	├── small_dataset.tar # 用于训练的小数据集 (10张图片)
	├── small_lfw.bin # 用于评估的小数据集 (20张图片)
├── docs/ # 存放 TIPC 测试数据的目录
	├── install.md # 安装 TIPC 所需环境的文档
	├── test_train_inference_python.md # 测试Linux上python训练预测的文档
	├── test_serving.md # 测试Linux上python serving预测的文档
├── prepare.sh                        # 完成test_*.sh运行所需要的数据和模型下载
├── test_serving.sh    # 测试python训练预测的主程序
├── test_train_inference_python.sh    # 测试python训练预测的主程序
├── common_func.sh                    # 通用shell脚本函数
└── readme.md                         # TIPC使用文档
```

### 测试流程
使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程如下：
<div align="center">
    <img src="docs/test.png" width="800">
</div>

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_*.sh`，产出log，由log可以看到不同配置是否运行成功；

其中，有1个测试主程序，功能如下：
- `test_train_inference_python.sh`：测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

<a name="more"></a>
#### 更多教程
各功能测试中涉及混合精度、裁剪、量化等训练相关，及mkldnn、Tensorrt等多种预测相关参数配置，请点击下方相应链接了解更多细节和使用教程：  
[test_train_inference_python 使用](docs/test_train_inference_python.md)  