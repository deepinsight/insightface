
## Subcenter ArcFace

### 1. Motivation

We introduce one extra hyperparameter (subcenter number `loss_K`) to ArcFace to relax the intra-class compactness constraint. In our experiments, we find ``loss_K=3`` can achieve a good balance between accuracy and robustness.

![difference](https://insightface.ai/assets/img/github/subcenterarcfacediff.png)

### 2. Implementation

The training process of Subcenter ArcFace is almost same as [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace)
The increased GPU memory consumption can be easily alleviated by our parallel framework.

![framework](https://insightface.ai/assets/img/github/subcenterarcfaceframework.png)

### 3. Training Dataset

1. MS1MV0 (The noise rate is around 50%), download link ([baidu drive](https://pan.baidu.com/s/1bSamN5CLiSrxOuGi-Lx7tw), code ``8ql0``)  ([dropbox](https://www.dropbox.com/sh/y2mj25uj440f7bl/AABc7pCJvUvxEcmXs8WYi9Zaa?dl=0))

### 4. Training Steps

1). Train Sub-center ArcFace (``loss_K=3``) on MS1MV0.

2). Drop non-dominant subcenters and high-confident noisy data (``>75 degrees``). 

  ``
  python drop.py --data <ms1mv0-path> --model <step-1-pretrained-model> --threshold 75 --k 3 --output <ms1mv0-drop75-path>
  ``
  
3). Train ArcFace on the new ``MS1MV0-Drop75`` dataset.

### 5. Pretrained Models and Logs
  [baidu drive](https://pan.baidu.com/s/1yikOW1Xzm1XIHu0uv0RdRw) code ``3jsh``. [gdrive](https://drive.google.com/file/d/1h8Ybz6mJ7n2IfLbDv2HUU37OdVHn7YPg/view?usp=sharing)

### Citation

If you find *Sub-center ArcFace* useful in your research, please consider to cite the following related papers:

```
@inproceedings{deng2020subcenter,
  title={Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces},
  author={Deng, Jiankang and Guo, Jia and Liu, Tongliang and Gong, Mingming and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on European Conference on Computer Vision},
  year={2020}
}
```

