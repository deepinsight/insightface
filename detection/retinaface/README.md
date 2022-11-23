# RetinaFace Face Detector

## Introduction

RetinaFace is a practical single-stage [SOTA](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) face detector which is initially introduced in [arXiv technical report](https://arxiv.org/abs/1905.00641) and then accepted by [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html).

![demoimg1](https://insightface.ai/assets/img/github/11513D05.jpg)

![demoimg2](https://insightface.ai/assets/img/github/widerfacevaltest.png)

## Data

1. Download our annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [gdrive](https://drive.google.com/file/d/1BbXxIiY-F74SumCNG6iwmJJ5K3heoemT/view?usp=sharing)

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
2. Download ImageNet pretrained models and put them into ``model/``(these models are not for detection testing/inferencing but training and parameters initialization). 

    ImageNet ResNet50 ([baidu cloud](https://pan.baidu.com/s/1WAkU9ZA_j-OmzO-sdk9whA) and [googledrive](https://drive.google.com/file/d/1ibQOCG4eJyTrlKAJdnioQ3tyGlnbSHjy/view?usp=sharing)). 

    ImageNet ResNet152 ([baidu cloud](https://pan.baidu.com/s/1nzQ6CzmdKFzg8bM8ChZFQg) and [googledrive](https://drive.google.com/file/d/1FEjeiIB4u-XBYdASgkyx78pFybrlKUA4/view?usp=sharing)).

3. Start training with ``CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --prefix ./model/retina --network resnet``.  
Before training, you can check the ``resnet`` network configuration (e.g. pretrained model path, anchor setting and learning rate policy etc..) in ``rcnn/config.py``.
4. We have two predefined network settings named ``resnet``(for medium and large models) and ``mnet``(for lightweight models).

## Testing

Please check ``test.py`` for testing.

## RetinaFace Pretrained Models

Pretrained Model: RetinaFace-R50 ([baidu cloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [googledrive](https://drive.google.com/file/d/1_DKgGxQWqlTqe78pw0KavId9BIMNUWfu/view?usp=sharing)) is a medium size model with ResNet50 backbone.
It can output face bounding boxes and five facial landmarks in a single forward pass.

WiderFace validation mAP: Easy 96.5, Medium 95.6, Hard 90.4. 

To avoid the confliction with the WiderFace Challenge (ICCV 2019), we postpone the release time of our best model.

## Third-party

[yangfly](https://github.com/yangfly): RetinaFace-MobileNet0.25 ([baidu cloud](https://pan.baidu.com/s/1P1ypO7VYUbNAezdvLm2m9w):nzof).
WiderFace validation mAP: Hard 82.5. (model size: 1.68Mb) 

[clancylian](https://github.com/clancylian/retinaface): C++ version

RetinaFace in [modelscope](https://modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary)

## References

```  
@inproceedings{Deng2020CVPR,
title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle = {CVPR},
year = {2020}
}
```


