
# InsightFace: 2D and 3D Face Analysis Project

<div align="left">
  <img src="https://insightface.ai/assets/img/custom/logo3.jpg" width="240"/>
</div>

InsightFace project is mainly maintained By [Jia Guo](mailto:guojia@gmail.com?subject=[GitHub]%20InsightFace%20Project) and [Jiankang Deng](https://jiankangdeng.github.io/). 

For all main contributors, please check [contributing](#contributing).

## Top News

**`2021-08-07`**: Add new [model_zoo](model_zoo) page.

**`2021-07-13`**: We now have implementations based on [paddlepaddle](https://github.com/PaddlePaddle): [arcface_paddle](recognition/arcface_paddle) for face recognition and [blazeface_paddle](detection/blazeface_paddle) for face detection.

**`2021-07-09`**: We add a [person_detection](examples/person_detection) example, trained by [SCRFD](detection/scrfd), which can be called directly by our [python-library](python-package).

**`2021-06-05`**: We launch a [Masked Face Recognition Challenge & Workshop](challenges/iccv21-mfr) on ICCV 2021.

**`2021-05-15`**: We released an efficient high accuracy face detection approach called [SCRFD](detection/scrfd).

**`2021-04-18`**: We achieved Rank-4th on NIST-FRVT 1:1, see [leaderboard](https://pages.nist.gov/frvt/html/frvt11.html).

**`2021-03-13`**: We have released our official ArcFace PyTorch implementation, see [here](recognition/arcface_torch). 

## License

The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

Both manual-downloading models from our github repo and auto-downloading models with our [python-library](python-package) follow the above license policy(which is for non-commercial research purposes only).



## Introduction

[InsightFace](https://insightface.ai) is an open source 2D&3D deep face analysis toolbox, mainly based on PyTorch and MXNet. 

Please check our [website](https://insightface.ai) for detail.

The master branch works with **PyTorch 1.6+** and/or **MXNet=1.6-1.8**, with **Python 3.x**.

InsightFace efficiently implements a rich variety of state of the art algorithms of face recognition, face detection and face alignment, which optimized for both training and deployment.


### ArcFace Video Demo


[<img src=https://insightface.ai/assets/img/github/facerecognitionfromvideo.PNG width="760" />](https://www.youtube.com/watch?v=y-D1tReryGA&t=81s)


Please click the image to watch the Youtube video. For Bilibili users, click [here](https://www.bilibili.com/video/av38041494?from=search&seid=11501833604850032313).



## Projects

The [page](https://insightface.ai/projects) on InsightFace website also describes all supported projects in InsightFace.

You may also interested in some [challenges](https://insightface.ai/challenges) hold by InsightFace.



## Face Recognition

### Introduction

In this module, we provide training data, network settings and loss designs for deep face recognition.

The supported methods are as follows:

- [x] [ArcFace_mxnet (CVPR'2019)](recognition/arcface_mxnet)
- [x] [ArcFace_torch (CVPR'2019)](recognition/arcface_torch)
- [x] [SubCenter ArcFace (ECCV'2020)](recognition/subcenter_arcface)
- [x] [PartialFC_mxnet (Arxiv'2020)](recognition/partial_fc)
- [x] [PartialFC_torch (Arxiv'2020)](recognition/arcface_torch)
- [x] [VPL (CVPR'2021)](recognition/vpl)
- [x] [OneFlow_face](recognition/oneflow_face)
- [x] [ArcFace_Paddle (CVPR'2019)](recognition/arcface_paddle)

Commonly used network backbones are included in most of the methods, such as IResNet, MobilefaceNet, MobileNet, InceptionResNet_v2, DenseNet, etc..


### Datasets

The training data includes, but not limited to the cleaned MS1M, VGG2 and CASIA-Webface datasets, which were already packed in MXNet binary format. Please [dataset](recognition/_dataset_) page for detail.

### Evaluation

We provide standard IJB and Megaface evaluation pipelines in [evaluation](recognition/_evaluation_)


### Pretrained Models

**Please check [Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) for more pretrained models.**

### Third-party Re-implementation of ArcFace

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

### Introduction

<div align="left">
  <img src="https://insightface.ai/assets/img/github/11513D05.jpg" width="640"/>
</div>

In this module, we provide training data with annotation, network settings and loss designs for face detection training, evaluation and inference.

The supported methods are as follows:

- [x] [RetinaFace (CVPR'2020)](detection/retinaface)
- [x] [SCRFD (Arxiv'2021)](detection/scrfd)
- [x] [blazeface_paddle](detection/blazeface_paddle)

[RetinaFace](detection/retinaface) is a practical single-stage face detector which is accepted by [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html). We provide training code, training dataset, pretrained models and evaluation scripts. 

[SCRFD](detection/scrfd) is an efficient high accuracy face detection approach which is initialy described in [Arxiv](https://arxiv.org/abs/2105.04714). We provide an easy-to-use pipeline to train high efficiency face detectors with NAS supporting.


## Face Alignment

### Introduction

<div align="left">
  <img src="https://insightface.ai/assets/img/custom/thumb_sdunet.png" width="600"/>
</div>

In this module, we provide datasets and training/inference pipelines for face alignment.

Supported methods:

- [x] [SDUNets (BMVC'2018)](alignment/heatmap)
- [x] [SimpleRegression](alignment/coordinate_reg)


[SDUNets](alignment/heatmap) is a heatmap based method which accepted on [BMVC](http://bmvc2018.org/contents/papers/0051.pdf).

[SimpleRegression](alignment/coordinate_reg) provides very lightweight facial landmark models with fast coordinate regression. The input of these models is loose cropped face image while the output is the direct landmark coordinates.


## Citation

If you find *InsightFace* useful in your research, please consider to cite the following related papers:

```

@article{guo2021sample,
  title={Sample and Computation Redistribution for Efficient Face Detection},
  author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2105.04714},
  year={2021}
}

@inproceedings{an2020partical_fc,
  title={Partial FC: Training 10 Million Identities on a Single Machine},
  author={An, Xiang and Zhu, Xuhan and Xiao, Yang and Wu, Lan and Zhang, Ming and Gao, Yuan and Qin, Bin and
  Zhang, Debing and Fu Ying},
  booktitle={Arxiv 2010.05222},
  year={2020}
}

@inproceedings{deng2020subcenter,
  title={Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces},
  author={Deng, Jiankang and Guo, Jia and Liu, Tongliang and Gong, Mingming and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on European Conference on Computer Vision},
  year={2020}
}

@inproceedings{Deng2020CVPR,
title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle = {CVPR},
year = {2020}
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

## Contributing

Main contributors:

- [Jia Guo](https://github.com/nttstar), ``guojia[at]gmail.com``
- [Jiankang Deng](https://github.com/jiankangdeng) ``jiankangdeng[at]gmail.com``
- [Xiang An](https://github.com/anxiangsir) ``anxiangsir[at]gmail.com``
- [Jack Yu](https://github.com/szad670401) ``jackyu961127[at]gmail.com``

