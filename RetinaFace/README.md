# RetinaFace Face Detector

## Introduction

RetinaFace is a practical single-stage face detector which is initially described in [arXiv technical report](https://arxiv.org/abs/1905.00641)

![demoimg1](https://github.com/deepinsight/insightface/blob/master/resources/11513D05.jpg)

![demoimg2](https://github.com/deepinsight/insightface/blob/master/resources/widerfacevaltest.png)

## Data

1. Download our annotations (face bounding boxes & five facial landmarks) from [baiducloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

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

## Training

Please check ``train.py`` for training.
1. Copy ``rcnn/sample_config.py`` to ``rcnn/config.py``
2. Download pretrained models and put them into ``model/``. TODO_LINK
3. Start training with ``CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --prefix ./model/retina --network resnet``.  You may want to check the ``resnet`` network configuration in ``rcnn/config.py`` before starting, like pretrained model path, anchor setting and learning rate policy etc..
4. Basically we have two predefined network settings called ``resnet``(for medium and large size models) and ``mnet``(for lightweight models).

## Testing

Please check ``test.py`` for testing.

## Models

Pretrained Model: RetinaFace-R50 ([baiducloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [dropbox](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0)) is a medium size model with ResNet50 backbone.
It can output face bounding boxes and five facial landmarks in a single forward pass.
WiderFace validation mAP: Easy 96.5, Medium 95.6, Hard 90.4. 

To avoid the confliction with the WiderFace Challenge (ICCV 2019), we postpone the release time of our best model.

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


