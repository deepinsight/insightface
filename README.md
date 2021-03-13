
# InsightFace: 2D and 3D Face Analysis Project

By [Jia Guo](mailto:guojia@gmail.com?subject=[GitHub]%20InsightFace%20Project) and [Jiankang Deng](https://jiankangdeng.github.io/)

## License

The code of InsightFace is released under the MIT License. There is no limitation for both acadmic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

## Introduction

InsightFace is an open source 2D&3D deep face analysis toolbox, mainly based on MXNet. 

The master branch works with **MXNet 1.2 to 1.6**, with **Python 3.x**.


 ## ArcFace Video Demo

[![ArcFace Demo](https://github.com/deepinsight/insightface/blob/master/resources/facerecognitionfromvideo.PNG)](https://www.youtube.com/watch?v=y-D1tReryGA&t=81s)

Please click the image to watch the Youtube video. For Bilibili users, click [here](https://www.bilibili.com/video/av38041494?from=search&seid=11501833604850032313).

## Recent Update

**`2021-03-13`**: We have released our official ArcFace PyTorch implementation, see [here](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). 

**`2021-03-09`**: [Tips](https://github.com/deepinsight/insightface/issues/1426) for training large-scale face recognition model, such as millions of IDs(classes).

**`2021-02-21`**: We provide a simple face mask renderer [here](https://github.com/deepinsight/insightface/tree/master/recognition/tools) which can be used as a data augmentation tool while training face recoginition models.

**`2021-01-20`**: [OneFlow](https://github.com/Oneflow-Inc/oneflow) based implementation of ArcFace and Partial-FC, [here](https://github.com/deepinsight/insightface/tree/master/recognition/oneflow_face).

**`2020-10-13`**: A new training method and one large training set(360K IDs) were released [here](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc) by DeepGlint.

**`2020-10-09`**: We opened a large scale recognition test benchmark [IFRT](https://github.com/deepinsight/insightface/tree/master/challenges/IFRT)

**`2020-08-01`**: We released lightweight facial landmark models with fast coordinate regression(106 points). See detail [here](https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg).

**`2020-04-27`**: InsightFace pretrained models and MS1M-Arcface are now specified as the only external training dataset, for iQIYI iCartoonFace challenge, see detail [here](http://challenge.ai.iqiyi.com/detail?raceId=5def71b4e9fcf68aef76a75e).

**`2020.02.21`**: Instant discussion group created on QQ with group-id: 711302608. For English developers, see install tutorial [here](https://github.com/deepinsight/insightface/issues/1069).

**`2020.02.16`**: RetinaFace now can detect faces with mask, for anti-CoVID19, see detail [here](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFaceAntiCov)

**`2019.08.10`**: We achieved 2nd place at [WIDER Face Detection Challenge 2019](http://wider-challenge.org/2019.html).

**`2019.05.30`**: [Presentation at cvmart](https://pan.baidu.com/s/1v9fFHBJ8Q9Kl9Z6GwhbY6A)

**`2019.04.30`**: Our Face detector ([RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace)) obtains state-of-the-art results on [the WiderFace dataset](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html).

**`2019.04.14`**: We will launch a [Light-weight Face Recognition challenge/workshop](https://github.com/deepinsight/insightface/tree/master/challenges/iccv19-lfr) on ICCV 2019.

**`2019.04.04`**: Arcface achieved state-of-the-art performance (7/109) on the NIST Face Recognition Vendor Test (FRVT) (1:1 verification)
[report](https://www.nist.gov/sites/default/files/documents/2019/04/04/frvt_report_2019_04_04.pdf) (name: Imperial-000 and Imperial-001). Our solution is based on [MS1MV2+DeepGlintAsian, ResNet100, ArcFace loss]. 

**`2019.02.08`**: Please check [https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace) for our parallel training code which can easily and efficiently support one million identities on a single machine (8* 1080ti).

**`2018.12.13`**: Inference acceleration [TVM-Benchmark](https://github.com/deepinsight/insightface/wiki/TVM-Benchmark).

**`2018.10.28`**: Light-weight attribute model [Gender-Age](https://github.com/deepinsight/insightface/tree/master/gender-age). About 1MB, 10ms on single CPU core. Gender accuracy 96% on validation set and 4.1 age MAE.

**`2018.10.16`**: We achieved state-of-the-art performance on [Trillionpairs](http://trillionpairs.deepglint.com/results) (name: nttstar) and [IQIYI_VID](http://challenge.ai.iqiyi.com/detail?raceId=5afc36639689443e8f815f9e) (name: WitcheR). 


## Contents
[Deep Face Recognition](#deep-face-recognition)
- [Introduction](#introduction)
- [Training Data](#training-data)
- [Train](#train)
- [Pretrained Models](#pretrained-models)
- [Verification Results On Combined Margin](#verification-results-on-combined-margin)
- [Test on MegaFace](#test-on-megaface)
- [512-D Feature Embedding](#512-d-feature-embedding)
- [Third-party Re-implementation](#third-party-re-implementation)

[Face Detection](#face-detection)
- [RetinaFace](#retinaface)
- [RetinaFaceAntiCov](#retinafaceanticov)

[Face Alignment](#face-alignment)
- [DenseUNet](#denseunet)
- [CoordinateReg](#coordinatereg)


[Citation](#citation)

[Contact](#contact)

## Deep Face Recognition

### Introduction

In this module, we provide training data, network settings and loss designs for deep face recognition.
The training data includes, but not limited to the cleaned MS1M, VGG2 and CASIA-Webface datasets, which were already packed in MXNet binary format.
The network backbones include ResNet, MobilefaceNet, MobileNet, InceptionResNet_v2, DenseNet, etc..
The loss functions include Softmax, SphereFace, CosineFace, ArcFace, Sub-Center ArcFace and Triplet (Euclidean/Angular) Loss.

You can check the detail page of our work [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace)(which accepted in CVPR-2019) and [SubCenter-ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/SubCenter-ArcFace)(which accepted in ECCV-2020).

![margin penalty for target logit](https://github.com/deepinsight/insightface/raw/master/resources/arcface.png)

Our method, ArcFace, was initially described in an [arXiv technical report](https://arxiv.org/abs/1801.07698). By using this module, you can simply achieve LFW 99.83%+ and Megaface 98%+ by a single model. This module can help researcher/engineer to develop deep face recognition algorithms quickly by only two steps: download the binary dataset and run the training script.

### Training Data

All face images are aligned by ficial five landmarks and cropped to 112x112:

Please check [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) for detail information and dataset downloading.


* Please check *src/data/face2rec2.py* on how to build a binary face dataset. You can either choose *MTCNN* or *RetinaFace* to align the faces.

### Train

1. Install `MXNet` with GPU support (Python 3.X).

```
pip install mxnet-cu101 # which should match your installed cuda version
```

2. Clone the InsightFace repository. We call the directory insightface as *`INSIGHTFACE_ROOT`*.

```
git clone --recursive https://github.com/deepinsight/insightface.git
```

3. Download the training set (`MS1M-Arcface`) and place it in *`$INSIGHTFACE_ROOT/recognition/datasets/`*. Each training dataset includes at least following 6 files:

```Shell
    faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

The first three files are the training dataset while the last three files are verification sets.

4. Train deep face recognition models.
In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/recognition/ArcFace/`*.

Place and edit config file:
```Shell
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```

We give some examples below. Our experiments were conducted on the Tesla P40 GPU.

(1). Train ArcFace with LResNet100E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore
```
It will output verification results of *LFW*, *CFP-FP* and *AgeDB-30* every 2000 batches. You can check all options in *config.py*.
This model can achieve *LFW 99.83+* and *MegaFace 98.3%+*.

(2). Train CosineFace with LResNet50E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore
```

(3). Train Softmax with LMobileNet-GAP.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss softmax --dataset emore
```

(4). Fine-turn the above Softmax model with Triplet loss.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss triplet --lr 0.005 --pretrained ./models/m1-softmax-emore,1
```

(5). Training in model parallel acceleration.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_parall.py --network r100 --loss arcface --dataset emore
```

5. Verification results.

*LResNet100E-IR* network trained on *MS1M-Arcface* dataset with ArcFace loss:

| Method  | LFW(%) | CFP-FP(%) | AgeDB-30(%) |  
| ------- | ------ | --------- | ----------- |  
|  Ours   | 99.80+ | 98.0+     | 98.20+      |   



### Pretrained Models

You can use `$INSIGHTFACE/src/eval/verification.py` to test all the pre-trained models.

**Please check [Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) for more pretrained models.**



### Verification Results on Combined Margin

A combined margin method was proposed as a function of target logits value and original `θ`:

```
COM(θ) = cos(m_1*θ+m_2) - m_3
```

For training with `m1=1.0, m2=0.3, m3=0.2`, run following command:
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss combined --dataset emore
```

Results by using ``MS1M-IBUG(MS1M-V1)``

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

Please check *`$INSIGHTFACE_ROOT/Evaluation/megaface/`* to evaluate the model accuracy on Megaface. All aligned images were already provided.


### 512-D Feature Embedding

In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/deploy/`*. The input face image should be generally centre cropped. We use *RNet+ONet* of *MTCNN* to further align the image before sending it to the feature embedding network.

1. Prepare a pre-trained model.
2. Put the model under *`$INSIGHTFACE_ROOT/models/`*. For example, *`$INSIGHTFACE_ROOT/models/model-r100-ii`*.
3. Run the test script *`$INSIGHTFACE_ROOT/deploy/test.py`*.

For single cropped face image(112x112), total inference time is only 17ms on our testing server(Intel E5-2660 @ 2.00GHz, Tesla M40, *LResNet34E-IR*).

### Third-party Re-implementation

- TensorFlow: [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
- TensorFlow: [tf-insightface](https://github.com/AIInAi/tf-insightface)
- TensorFlow:[insightface](https://github.com/Fei-Wang/insightface)
- PyTorch: [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- PyTorch: [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
- Caffe: [arcface-caffe](https://github.com/xialuxi/arcface-caffe)
- Caffe: [CombinedMargin-caffe](https://github.com/gehaocool/CombinedMargin-caffe)
- Tensorflow: [InsightFace-tensorflow](https://github.com/luckycallor/InsightFace-tensorflow)
- TensorRT: [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)

## Face Detection

### RetinaFace

  RetinaFace is a practical single-stage [SOTA](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) face detector which is initially introduced in [arXiv technical report](https://arxiv.org/abs/1905.00641) and then accepted by [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html). We provide training code, training dataset, pretrained models and evaluation scripts. 
  
  ![demoimg1](https://github.com/deepinsight/insightface/blob/master/resources/11513D05.jpg)
  
  Please check [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace) for detail.
  
### RetinaFaceAntiCov 
 
  RetinaFaceAntiCov is an experimental module to identify face boxes with masks. Please check [RetinaFaceAntiCov](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFaceAntiCov) for detail.
  
  ![demoimg1](https://github.com/deepinsight/insightface/blob/master/resources/cov_test.jpg)


## Face Alignment

### DenseUNet

  Please check the [Menpo](https://github.com/jiankangdeng/MenpoBenchmark) Benchmark and our [Dense U-Net](https://github.com/deepinsight/insightface/tree/master/alignment/heatmapReg) for detail. We also provide other network settings such as classic hourglass. You can find all of training code, training dataset and evaluation scripts there.
  
### CoordinateReg

  On the other hand, in contrast to heatmap based approaches, we provide some lightweight facial landmark models with fast coordinate regression. The input of these models is loose cropped face image while the output is the direct landmark coordinates. See detail at [alignment-coordinateReg](https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg). Now only pretrained models available.
  
 <div align="center">
	<img src="https://github.com/nttstar/insightface-resources/blob/master/alignment/images/t1_out.jpg" alt="imagevis" width="800">
</div>


<div align="center">
	<img src="https://github.com/nttstar/insightface-resources/blob/master/alignment/images/C_jiaguo.gif" alt="videovis" width="240">
</div>


## Citation

If you find *InsightFace* useful in your research, please consider to cite the following related papers:

```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}

@inproceedings{guo2018stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={BMVC},
  year={2018}
}

@article{deng2018menpo,
  title={The Menpo benchmark for multi-pose 2D and 3D facial landmark localisation and tracking},
  author={Deng, Jiankang and Roussos, Anastasios and Chrysos, Grigorios and Ververas, Evangelos and Kotsia, Irene and Shen, Jie and Zafeiriou, Stefanos},
  journal={IJCV},
  year={2018}
}

@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```

## Contact

```
[Jia Guo](guojia[at]gmail.com)
[Jiankang Deng](jiankangdeng[at]gmail.com)
```
