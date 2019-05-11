# RetinaFace Face Detector

## Introduction

RetinaFace is a practical single-stage [SOTA](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) face detector which is initially described in [arXiv technical report](https://arxiv.org/abs/1905.00641)

![demoimg1](https://github.com/deepinsight/insightface/blob/master/resources/11513D05.jpg)

![demoimg2](https://github.com/deepinsight/insightface/blob/master/resources/widerfacevaltest.png)

## Data

1. Download our annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

2. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

3. Organise the dataset directory under ``insightface/RetinaFace/`` as follows:

```Shell
  data/retinaface/
    train/
      images/
      label.txt
    val/
      images/
      label.txt
    test/
      images/
      label.txt
```

## Install

1. Install MXNet with GPU support.
2. Install Deformable Convolution V2 operator from [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets) if you use the DCN based backbone.
3. Type ``make`` to build cxx tools.

## Training

Please check ``train.py`` for training.

1. Copy ``rcnn/sample_config.py`` to ``rcnn/config.py``
2. Download pretrained models and put them into ``model/``. 

    ImageNet ResNet50 ([baidu cloud](https://pan.baidu.com/s/1WAkU9ZA_j-OmzO-sdk9whA) and [dropbox](https://www.dropbox.com/s/48b850vmnaaasfl/imagenet-resnet-50.zip?dl=0)). 

    ImageNet ResNet152 ([baidu cloud](https://pan.baidu.com/s/1nzQ6CzmdKFzg8bM8ChZFQg) and [dropbox](https://www.dropbox.com/s/8ypcra4nqvm32v6/imagenet-resnet-152.zip?dl=0)).

3. Start training with ``CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --prefix ./model/retina --network resnet``.  
Before training, you can check the ``resnet`` network configuration (e.g. pretrained model path, anchor setting and learning rate policy etc..) in ``rcnn/config.py``.
4. We have two predefined network settings named ``resnet``(for medium and large models) and ``mnet``(for lightweight models).

## Testing

Please check ``test.py`` for testing.

## Models

Pretrained Model: RetinaFace-R50 ([baidu cloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [dropbox](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0)) is a medium size model with ResNet50 backbone.
It can output face bounding boxes and five facial landmarks in a single forward pass.

WiderFace validation mAP: Easy 96.5, Medium 95.6, Hard 90.4. 

To avoid the confliction with the WiderFace Challenge (ICCV 2019), we postpone the release time of our best model.

## Third-party Models

[yangfly](https://github.com/yangfly): RetinaFace-MobileNet0.25 ([baidu cloud](https://pan.baidu.com/s/1P1ypO7VYUbNAezdvLm2m9w)).

WiderFace validation mAP: Hard 82.5. (model size: 1.68Mb) 

## References

```
@inproceedings{yang2016wider,
title = {WIDER FACE: A Face Detection Benchmark},
author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
booktitle = {CVPR},
year = {2016}
}
  
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}
```


