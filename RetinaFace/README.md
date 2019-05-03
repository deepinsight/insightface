# RetinaFace Face Detector

## Introduction

RetinaFace is a robust single stage face detector which initially described as an [arXiv technical report](https://arxiv.org/abs/1905.00641)

![demoimg](https://github.com/deepinsight/insightface/blob/master/resources/11513D05.jpg)


## Training

1. Download groundtruth labels from [baiducloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or dropbox and organise the dataset directory under ``insightface/RetinaFace/`` as follows(images can be downloaded from WiderFace website directly):
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

## Testing

Please check ``test.py`` for model usage.

Pretrained Model: [RetinaFace-R50](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) is a medium size model with ResNet50 backbone. WiderFace validation mAP: Easy 96.5, Medium 95.6, Hard 90.4. It can output face bounding boxes and five landmarks in a single forward pass.



