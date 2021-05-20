# RetinaFace Anti Cov Face Detector

## Introduction

RetinaFace-Anti-Cov is a customized one stage face detector to help people protect themselves from CovID-19.

![demoimg1](https://github.com/deepinsight/insightface/blob/master/resources/cov_test.jpg)


## Testing

Please check ``test.py`` for testing.

Make sure that you set ``network='net3l'`` instead of ``'net3'`` for 'mnet_cov2' model, otherwise you will get incorrect landmarks.

## Pretrained Models

~~MobileNet0.25([baidu cloud](https://pan.baidu.com/s/1p8n4R2W-9WmmBWxYQEFcWg),code: fmfm)~~

Better: MobileNet0.25 ([baidu cloud](https://pan.baidu.com/s/16ihzPxjTObdbv0D6P6LmEQ), code: j3b6, [dropbox](https://www.dropbox.com/s/6rhhxsbh2qik65k/cov2.zip?dl=0))



## References

```
  
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}
```


