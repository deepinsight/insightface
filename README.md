
# InsightFace: 2D and 3D Face Analysis Project

<div align="left">
  <img src="https://insightface.ai/assets/img/custom/logo3.jpg" width="240"/>
</div>

InsightFace project is mainly maintained By [Jia Guo](mailto:guojia@gmail.com?subject=[GitHub]%20InsightFace%20Project) and [Jiankang Deng](https://jiankangdeng.github.io/). 

For all main contributors, please check [contributing](#contributing).

## License

The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

Both manual-downloading models from our github repo and auto-downloading models with our [python-library](python-package) follow the above license policy(which is for non-commercial research purposes only).

## Top News

**`2023-08-08`**: We released the implementation of [Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views](https://arxiv.org/abs/2212.02997) at [reconstruction/gaze](reconstruction/gaze).

**`2023-05-03`**: We have launched the ongoing version of wild face anti-spoofing challenge. See details [here](https://github.com/deepinsight/insightface/tree/master/challenges/cvpr23-fas-wild#updates).

**`2023-04-01`**: We move the swapping demo to Discord bot, which support editing on Midjourney generated images, see detail at [web-demos/swapping_discord](web-demos/swapping_discord).

**`2023-02-13`**: We launch a large scale in the wild face anti-spoofing challenge on CVPR23 Workshop, see details at [challenges/cvpr23-fas-wild](challenges/cvpr23-fas-wild).

**`2022-11-28`**: Single line code for facial identity swapping in our python packge ver 0.7, please check the example [here](examples/in_swapper).

**`2022-10-28`**: [MFR-Ongoing](http://iccv21-mfr.com) website is refactored, please create issues if there's any bug.

**`2022-09-22`**: Now we have [web-demos](web-demos): [face-localization](http://demo.insightface.ai:7007/), [face-recognition](http://demo.insightface.ai:7008/), and [face-swapping](http://demo.insightface.ai:7009/).

**`2022-08-12`**: We achieved Rank-1st of 
[Perspective Projection Based Monocular 3D Face Reconstruction Challenge](https://tianchi.aliyun.com/competition/entrance/531961/introduction)
of [ECCV-2022 WCPA Workshop](https://sites.google.com/view/wcpa2022), [paper](https://arxiv.org/abs/2208.07142) and [code](reconstruction/jmlr).

**`2022-03-30`**: [Partial FC](https://arxiv.org/abs/2203.15565) accepted by CVPR-2022.

**`2022-02-23`**: [SCRFD](detection/scrfd) accepted by [ICLR-2022](https://iclr.cc/Conferences/2022).

**`2021-11-30`**: [MFR-Ongoing](challenges/mfr) challenge launched(same with IFRT), which is an extended version of [iccv21-mfr](challenges/iccv21-mfr).

**`2021-10-29`**: We achieved 1st place on the [VISA track](https://pages.nist.gov/frvt/plots/11/visa.html) of [NIST-FRVT 1:1](https://pages.nist.gov/frvt/html/frvt11.html) by using Partial FC (Xiang An, Jiankang Deng, Jia Guo).

**`2021-10-11`**: [Leaderboard](https://insightface.ai/mfr21) of [ICCV21 - Masked Face Recognition Challenge](challenges/iccv21-mfr) released. Video: [Youtube](https://www.youtube.com/watch?v=lL-7l5t6x2w), [Bilibili](https://www.bilibili.com/video/BV15b4y1h79N/).

**`2021-06-05`**: We launch a [Masked Face Recognition Challenge & Workshop](challenges/iccv21-mfr) on ICCV 2021.



## Introduction

[InsightFace](https://insightface.ai) is an open source 2D&3D deep face analysis toolbox, mainly based on PyTorch and MXNet. 

Please check our [website](https://insightface.ai) for detail.

The master branch works with **PyTorch 1.6+** and/or **MXNet=1.6-1.8**, with **Python 3.x**.

InsightFace efficiently implements a rich variety of state of the art algorithms of face recognition, face detection and face alignment, which optimized for both training and deployment.

## Quick Start

Please start with our [python-package](python-package/), for testing detection, recognition and alignment models on input images.


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
- [x] [PartialFC_mxnet (CVPR'2022)](recognition/partial_fc)
- [x] [PartialFC_torch (CVPR'2022)](recognition/arcface_torch)
- [x] [VPL (CVPR'2021)](recognition/vpl)
- [x] [Arcface_oneflow](recognition/arcface_oneflow)
- [x] [ArcFace_Paddle (CVPR'2019)](recognition/arcface_paddle)

Commonly used network backbones are included in most of the methods, such as IResNet, MobilefaceNet, MobileNet, InceptionResNet_v2, DenseNet, etc..


### Datasets

The training data includes, but not limited to the cleaned MS1M, VGG2 and CASIA-Webface datasets, which were already packed in MXNet binary format. Please [dataset](recognition/_datasets_) page for detail.

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
- TensorRT: [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST)
- ONNXRuntime C++: [ArcFace-ONNXRuntime](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/glint_arcface.cpp)
- ONNXRuntime Go: [arcface-go](https://github.com/jack139/arcface-go)
- MNN: [ArcFace-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_glint_arcface.cpp)
- TNN: [ArcFace-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_glint_arcface.cpp)
- NCNN: [ArcFace-NCNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ncnn/cv/ncnn_glint_arcface.cpp)

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
@inproceedings{ren2023pbidr,
  title={Facial Geometric Detail Recovery via Implicit Representation},
  author={Ren, Xingyu and Lattas, Alexandros and Gecer, Baris and Deng, Jiankang and Ma, Chao and Yang, Xiaokang},
  booktitle={2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG)},  
  year={2023}
 }

@article{guo2021sample,
  title={Sample and Computation Redistribution for Efficient Face Detection},
  author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2105.04714},
  year={2021}
}

@inproceedings{gecer2021ostec,
  title={OSTeC: One-Shot Texture Completion},
  author={Gecer, Baris and Deng, Jiankang and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
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
- [Baris Gecer](https://barisgecer.github.io/) ``barisgecer[at]msn.com``
