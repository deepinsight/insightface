# Arcface-Paddle

本部分主要包含人脸检测+人脸识别+系统串联三个部分。整体目录结构如下。


* [1. Face detection using BlazeFace](./det)
* [2. Face recognition using arcface](./rec)
* [3. End-to-end inference using PaddlePaddle](./system)


在人脸检测任务中，在WiderFace数据集上，BlazeFace的速度与精度指标信息如下。

| 模型结构                  | 模型大小 | WiderFace精度   | CPU 耗时 | GPU 耗时 |
| ------------------------- | ----- | ----- | -------- | -------- |
| BlazeFace-FPN-SSH      | 0.65MB | 0.907/0.883/0.793 | 87ms  | 40ms  | 40ms |
| RetinaFace      | 1.68MB | -/-/0.825 | 182ms  | 42ms |

在人脸识别任务中，基于MS1M训练集，模型指标在lfw、cfp_fp、agedb30上的精度指标以及CPU、GPU的预测耗时如下。

| 模型结构                  | lfw   | cfp_fp | agedb30 | CPU 耗时 | GPU 耗时 |
| ------------------------- | ----- | ------ | ------- | -------- | -------- |
| MobileFaceNet-Paddle      | 0.9945 | 0.9343  | 0.9613   | 38.84ms  | 2.26ms   |
| MobileFaceNet-insightface | 0.9950 | 0.8894  | 0.9591   | 7.32ms   | 4.71ms   |


**测试环境：**
* CPU: Intel(R) Xeon(R) Gold 6184 CPU @ 2.40GHz
* GPU: a single NVIDIA Tesla V100

基于少量老友记人脸数据库，轻量级模型的识别结果可视化如下。

* todo
