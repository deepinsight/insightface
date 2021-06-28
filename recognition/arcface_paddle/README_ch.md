简体中文 | [English](README_en.md)

# Arcface-Paddle

本部分主要包含人脸检测+人脸识别+系统串联三个部分。整体目录结构如下。


* [1. Face detection using BlazeFace](./det/README_ch.md)
* [2. Face recognition using arcface](./rec/README_ch.md)
* [3. End-to-end inference using PaddlePaddle](./system/README_ch.md)


Note: 在此非常感谢 [GuoQuanhao](https://github.com/GuoQuanhao) 基于PaddlePaddle复现了 [Arcface的基线模型](https://github.com/GuoQuanhao/arcface-Paddle)。


在人脸检测任务中，在WiderFace数据集上，BlazeFace的速度与精度指标信息如下。

| 模型结构                  | 模型大小 | WiderFace精度   | CPU 耗时 | GPU 耗时 |
| ------------------------- | ----- | ----- | -------- | -------- |
| BlazeFace-FPN-SSH      | 0.65MB | 0.907/0.883/0.793 | 55.0ms  |  6.2ms |
| RetinaFace      | 1.68MB | -/-/0.825 | 182.0ms  | 17.4ms |

在人脸识别任务中，基于MS1M训练集，模型指标在lfw、cfp_fp、agedb30上的精度指标以及CPU、GPU的预测耗时如下。

| 模型结构                  | lfw   | cfp_fp | agedb30  | GPU 耗时 |
| ------------------------- | ----- | ------ | ------- | -------- |
| MobileFaceNet-Paddle      | 0.9945 | 0.9343  | 0.9613  | 2.3ms   |
| MobileFaceNet-mxnet | 0.9950 | 0.8894  | 0.9591  | 4.7ms   |


**测试环境：**
* CPU: Intel(R) Xeon(R) Gold 6184 CPU @ 2.40GHz
* GPU: a single NVIDIA Tesla V100

基于少量老友记人脸数据库，轻量级模型的识别结果可视化如下。

* coming soon!
