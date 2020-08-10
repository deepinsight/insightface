
## SubCenter ArcFace

### Training Configuration

The training process of SubCenter-ArcFace is almost the same with [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace), except for one extra configuration item `loss_K` which is commonly set as ``3``.

### Dataset

1. MS1MV0(Full MS1M), download link ([baidulink](https://pan.baidu.com/s/1bSamN5CLiSrxOuGi-Lx7tw), code ``8ql0``)  ([googledrive](TODO))

### Training Steps

1. Train SubCenter-ArcFace with ``loss_K==3`` on MS1MV0.

2. Drop MS1MV0 on ``>75 degrees`` samples. 

  ``
  python drop.py --data <ms1mv0-path> --model <step-1-pretrained-model> --threshold 75 --k 3 --output <ms1mv0-drop75-path>
  ``
  
3. Train ArcFace on the new ``MS1MV0-Drop75`` dataset.
  

### Citation


