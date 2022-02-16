


## 1. Download Datasets and Unzip

Download WebFace42M from [https://www.face-benchmark.org/download.html](https://www.face-benchmark.org/download.html).


## 2. Create **Pre-shuffle** Rec File for DALI

Note: preshuffled rec is very important to DALI, and rec without preshuffled can cause performance degradation, origin insightface style rec file 
do not support Nvidia DALI, you must follow this command [mxnet.tools.im2rec](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) to generate a pre-shuffle rec file.

```shell
# 1) create train.lst using follow command
python -m mxnet.tools.im2rec --list --recursive train "Your WebFace42M Root"

# 2) create train.rec and train.idx using train.lst using following command
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 train "Your WebFace42M Root"
```

Finally, you will get three files: `train.lst`, `train.rec`, `train.idx`. which `train.idx`, `train.rec` are using for training.
