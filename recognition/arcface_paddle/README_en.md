[简体中文](README_ch.md) | English

# Arcface-Paddle

This tutorial contains 3 parts: (1) face detection. (2) face recognition (3) end-to-end system inference.

* [1. Face detection using BlazeFace](./det/README_en.md)
* [2. Face recognition using arcface](./rec/README_en.md)
* [3. End-to-end inference using PaddlePaddle](./system/README_en.md)


Note: Many thanks to [GuoQuanhao](https://github.com/GuoQuanhao) for the reproduction of the [Arcface basline using PaddlePaddle](https://github.com/GuoQuanhao/arcface-Paddle).


For face detection task, on WiderFace dataset, the following table shows mAP, speed and time cost for BlazeFace.


| Model structure                  | Model size | WiderFace mAP   | CPU time cost | GPU time cost |
| ------------------------- | ----- | ----- | -------- | -------- |
| BlazeFace-FPN-SSH      | 0.65MB | 0.907/0.883/0.793 | 25.4ms  |  5.6ms |
| RetinaFace      | 1.68MB | -/-/0.825 | 182.0ms  | 17.4ms |


For face recognition task, on MSAM dataset, the following table shows precision, speed and time cost for MobileFaceNet.


| Model structure           | lfw   | cfp_fp | agedb30  | GPU time cost |
| ------------------------- | ----- | ------ | ------- | -------- |
| MobileFaceNet-Paddle      | 0.9945 | 0.9343  | 0.9613  | 2.3ms   |
| MobileFaceNet-mxnet | 0.9950 | 0.8894  | 0.9591  | 4.7ms   |


**Benchmark environment:**
* CPU: Intel(R) Xeon(R) Gold 6184 CPU @ 2.40GHz
* GPU: a single NVIDIA Tesla V100


The recognition results of the light-weight model based on "Friends" can be seen as follows.


* coming soon!
