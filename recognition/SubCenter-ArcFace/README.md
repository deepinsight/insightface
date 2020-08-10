
## Subcenter ArcFace

### 1. Main Contribution

The training process of Subcenter ArcFace is almost same as [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace), except for one extra hyperparameter (subcenter number `loss_K`) to relax the intra-class compactness constraint. In our experiments, we find ``loss_K=3`` can achieve a good balance between accuracy and robustness.

![diff](https://github.com/deepinsight/insightface/blob/master/resources/subcenterarcface.png)![results](https://github.com/deepinsight/insightface/blob/master/resources/subcenterarcfacemulticenter.png)

![framework](https://github.com/deepinsight/insightface/blob/master/resources/subcenterarcfaceframework.png)

### 2. Training Dataset

1. MS1MV0 (The noise rate is around 50%), download link ([baidulink](https://pan.baidu.com/s/1bSamN5CLiSrxOuGi-Lx7tw), code ``8ql0``)  ([googledrive](TODO))

### 3. Training Steps

1). Train Sub-center ArcFace (``loss_K=3``) on MS1MV0.

2). Drop non-dominant subcenters and high-confident noisy data (``>75 degrees``). 

  ``
  python drop.py --data <ms1mv0-path> --model <step-1-pretrained-model> --threshold 75 --k 3 --output <ms1mv0-drop75-path>
  ``
  
3). Train ArcFace on the new ``MS1MV0-Drop75`` dataset.
  

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

