
# InsightFace: 2D and 3D Face Analysis Project

By Jia Guo and [Jiankang Deng](https://jiankangdeng.github.io/)

## License

The code of InsightFace is released under the MIT License.

## Recent Update

**`2018.05.16`**: A new training dataset released [here](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ) which can easily achieve much better accuracy. See [discussion](https://github.com/deepinsight/insightface/issues/215) for detail.

**`2018.04.23`**: Our implementation of [MobileFaceNet](https://arxiv.org/abs/1804.07573) is now available. Please set `--network y1` to use this lightweight but powerful backbone.

**`2018.03.26`**: We can train with combined margin(loss-type=5), see [Verification Results On Combined Margin](#verification-results-on-combined-margin).

**`2018.02.13`**: We achieved state-of-the-art performance on [MegaFace-Challenge](http://megaface.cs.washington.edu/results/facescrub.html). Please check our paper and code for implementation details.

## Contents
[Deep Face Recognition](#deep-face-recognition)
- [Introduction](#introduction)
- [Training Data](#training-Data)
- [Train](#train)
- [Pretrained Models](#pretrained-models)
- [Verification Results On Combined Margin](#verification-results-on-combined-margin)
- [Test on MegaFace](#test-on-megaface)
- [512-D Feature Embedding](#512-d-feature-embedding)
- [Third-party Re-implementation](#third-party-re-implementation)

[Face Alignment](#face-alignment)

[Face Detection](#face-detection)

[Citation](#citation)

[Contact](#contact)

## Deep Face Recognition

### Introduction

In this repository, we provide training data, network settings and loss designs for deep face recognition.
The training data includes the normalised MS1M and VGG2 datasets, which were already packed in the MxNet binary format.
The network backbones include ResNet, InceptionResNet_v2, DenseNet, DPN and MobiletNet.
The loss functions include Softmax, SphereFace, CosineFace, ArcFace and Triplet (Euclidean/Angular) Loss.
* loss-type=0:  Softmax
* loss-type=1:  SphereFace
* loss-type=2:  CosineFace
* loss-type=4:  ArcFace
* loss-type=5:  Combined Margin
* loss-type=12: TripletLoss

![margin penalty for target logit](https://github.com/deepinsight/insightface/raw/master/resources/arcface.png)

Our method, ArcFace, was initially described in an [arXiv technical report](https://arxiv.org/abs/1801.07698). By using this repository, you can simply achieve LFW 99.80%+ and Megaface 98%+ by a single model. This repository can help researcher/engineer to develop deep face recognition algorithms quickly by only two steps: download the binary dataset and run the training script.

### Training Data

All face images are aligned by [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) and cropped to 112x112:

* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://drive.google.com/open?id=1KORwx_DWyIScAjD6vbo4CSRu048APoum)
* Please check *src/data/face2rec2.py* on how to build a binary face dataset. Any public available *MTCNN* can be used to align the faces, and the performance should not change. We will improve the face normalisation step by full pose alignment methods recently.

**Note:** If you use the refined [MS1M](https://arxiv.org/abs/1607.08221) dataset and the cropped [VGG2](https://arxiv.org/abs/1710.08092) dataset, please cite the original papers.

### Train

1. Install `MXNet` with GPU support (Python 2.7).

```
pip install mxnet-cu80
```

2. Clone the InsightFace repository. We call the directory insightface as *`INSIGHTFACE_ROOT`*.

```
git clone --recursive https://github.com/deepinsight/insightface.git
```

3. Download the training set (`MS1M`) and place it in *`$INSIGHTFACE_ROOT/datasets/`*. Each training dataset includes following 7 files:

```Shell
    faces_ms1m_112x112/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_ff.bin
       cfp_fp.bin
       agedb_30.bin
```

The first three files are the training dataset while the last four files are verification sets.

4. Train deep face recognition models.
In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/src/`*.
```
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
```
We give some examples below. Our experiments were conducted on the Tesla P40 GPU.

(1). Train ArcFace with LResNet100E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 --data-dir ../datasets/faces_ms1m_112x112  --prefix ../model-r100
```
It will output verification results of *LFW*, *CFP-FF*, *CFP-FP* and *AgeDB-30* every 2000 batches. You can check all command line options in *train\_softmax.py*.
This model can achieve *LFW 99.80+* and *MegaFace 98.0%+*.

(2). Train CosineFace with LResNet50E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r50 --loss-type 2 --margin-m 0.35 --data-dir ../datasets/faces_ms1m_112x112 --prefix ../model-r50-amsoftmax
```

(3). Train Softmax with LMobileNetE.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network m1 --loss-type 0 --data-dir ../datasets/faces_ms1m_112x112 --prefix ../model-m1-softmax
```

(4). Fine-turn the above Softmax model with Triplet loss.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network m1 --loss-type 12 --lr 0.005 --mom 0.0 --per-batch-size 150 --data-dir ../datasets/faces_ms1m_112x112 --pretrained ../model-m1-softmax,50 --prefix ../model-m1-triplet
```

(5). Train LDPN107E network with Softmax loss on VGGFace2 dataset.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_softmax.py --network p107 --loss-type 0 --per-batch-size 64 --data-dir ../datasets/faces_vgg_112x112 --prefix ../model-p107-softmax
```

5. Verification results.

*LResNet100E-IR* network trained on *MS1M* dataset with ArcFace loss:

| Method  | LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) |  
| ------- | ------ | --------- | --------- | ----------- |  
|  Ours   | 99.80+ | 99.85+    | 94.0+     | 97.90+      |   

*LResNet50E-IR* network trained on *VGGFace2* dataset with ArcFace loss:

| Method  | LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) |
| ------- | ------ | --------- | --------- | ----------- |  
|  Ours   | 99.7+  |  99.6+    |   97.1+   |   95.7+     |  

We report the verification accuracy after removing training set overlaps to strictly follow the evaluation metric. `(C) means after cleaning`

| Dataset  | Identities | Images  | Identites(C) | Images(C) | Acc   | Acc(C) |
| -------- | ---------- | ------- | ------------ | --------- | ----- | ------ |
| LFW      | 85742      | 3850179 | 80995        | 3586128   | 99.83 | 99.81  |
| CFP-FP   | 85742      | 3850179 | 83706        | 3736338   | 94.04 | 94.03  |
| AgeDB-30 | 85742      | 3850179 | 83775        | 3761329   | 98.08 | 97.87  |

### Pretrained Models

You can use `$INSIGHTFACE/src/eval/verification.py` to test all the pre-trained models.

1. [LResNet50E-IR@BaiduDrive](https://pan.baidu.com/s/1mj6X7MK), [@GoogleDrive](https://drive.google.com/open?id=1x0-EiYX9jMUKiq-n1Bd9OCK4fVB3a54v)
Performance:

| Method  | LFW(%)     | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | MegaFace(%)   |
| ------- | ------     | --------- | --------- | ----------- | ------------- |
|  Ours   | 99.80      | 99.83     | 92.74     | 97.76       | 97.64         |

2. [LResNet34E-IR@BaiduDrive](https://pan.baidu.com/s/1jKahEXw)
Performance:

| Method  | LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | MegaFace(%)   |
| ------- | ------ | --------- | --------- | ----------- | ------------- |
|  Ours   | 99.65  | 99.77     | 92.12     | 97.70       | 96.70         |


*`Caffe`* [LResNet50E-IR@BaiduDrive](https://pan.baidu.com/s/1ENjcACInLfBGHZ8e7Fc-XA), converted by above MXNet model.
Performance:

| Method  | LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | MegaFace1M(%) |
| ------- | ------ | --------- | --------- | ----------- | ------------- |
|  Ours   | 99.74  | -TBD-     | -TBD-     | -TBD-       | -TBD-         |


### Verification Results on Combined Margin

A combined margin method was proposed as a function of target logits value and original `θ`:

```
COM(θ) = cos(m_1*θ+m_2) - m_3
```

For training with `m1=0.9, m2=0.4, m3=0.15`, run following command:
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss-type 5 --margin-a 0.9 --margin-m 0.4 --margin-b 0.15 --data-dir ../datasets/faces_ms1m_112x112  --prefix ../model-r100
```

| Method           | m1   | m2   | m3   | LFW   | CFP-FP | AgeDB-30 |
| ---------------- | ---- | ---- | ---- | ----- | ------ | -------- |
| W&F Norm Softmax | 1    | 0    | 0    | 99.28 | 88.50  | 95.13    |
| SphereFace       | 1.5  | 0    | 0    | 99.76 | 94.17  | 97.30    |
| CosineFace       | 1    | 0    | 0.35 | 99.80 | 94.4   | 97.91    |
| ArcFace          | 1    | 0.5  | 0    | 99.83 | 94.04  | 98.08    |
| Combined Margin  | 1.2  | 0.4  | 0    | 99.80 | 94.08  | 98.05    |
| Combined Margin  | 1.1  | 0    | 0.35 | 99.81 | 94.50  | 98.08    |
| Combined Margin  | 1    | 0.3  | 0.2  | 99.83 | 94.51  | 98.13    |
| Combined Margin  | 0.9  | 0.4  | 0.15 | 99.83 | 94.20  | 98.16    |

### Test on MegaFace

In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/src/megaface/`*.

**Note:** We found there are overlap identities between facescrub dataset and Megaface distractors, which significantly affects the identification performance. This list is released under *`$INSIGHTFACE_ROOT/src/megaface/`*.

1. Align all face images of facescrub dataset and megaface distractors. Please check the alignment scripts under *`$INSIGHTFACE_ROOT/src/align/`*.
2. Generate feature files for both facescrub and megaface images.
```
python -u gen_megaface.py
```
3. Remove Megaface noises which generates new feature files.
```
python -u remove_noises.py
```
4. Run megaface development kit to produce final result.

### 512-D Feature Embedding

In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/deploy/`*. The input face image should be generally centre cropped. We use *RNet+ONet* of *MTCNN* to further align the image before sending it to the feature embedding network.

1. Prepare a pre-trained model.
2. Put the model under *`$INSIGHTFACE_ROOT/models/`*. For example, *`$INSIGHTFACE_ROOT/models/model-r34-amf`*.
3. Run the test script *`$INSIGHTFACE_ROOT/deploy/test.py`*.

For single cropped face image(112x112), total inference time is only 17ms on our testing server(Intel E5-2660 @ 2.00GHz, Tesla M40, *LResNet34E-IR*).

### Third-party Re-implementation

- TensorFlow: [InsightFace_TF](https://github.com/auroua/InsightFace_TF)

## Face Alignment

Todo

## Face Detection

Todo

## Citation

If you find *InsightFace* useful in your research, please consider to cite the following related papers:

```
@article{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Zafeiriou, Stefanos},
journal={arXiv:1801.07698},
year={2018}
}
```

## Contact

```
[Jia Guo](guojia[at]gmail.com)
[Jiankang Deng](jiankangdeng[at]gmail.com)
```
