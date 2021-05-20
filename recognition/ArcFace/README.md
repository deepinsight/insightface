## ArcFace with Parallel Acceleration on both Features and Centers

### Memory Consumption and Training Speed

![Memoryspeed](https://github.com/deepinsight/insightface/blob/master/resources/memoryspeed.png)

Parallel acceleration on both feature x and centre W. Setting: ResNet 50, batch size 8 * 64, feature dimension 512, float point 32, GPU 8 * P40 (24GB).

### Illustration of Main Steps

![Memoryspeed](https://github.com/deepinsight/insightface/blob/master/resources/mainsteps.png)

Parallel calculation by simple matrix partition. Setting: ResNet 50, batch size 8 * 64, feature dimension 512, float point 32, identity number 1 Million, GPU 8 * 1080ti (11GB). Communication cost: 1MB (feature x). Training speed: 800 samples/second.

**Note:** Replace ``train.py`` with ``train_parall.py`` in following examples if you want to use parallel acceleration.

### Model Training

1. Install `MXNet` with GPU support.

```
pip install mxnet-cu100 # mxnet-cu102
```

2. Clone the InsightFace repository. We call the directory insightface as *`INSIGHTFACE_ROOT`*.

```
git clone --recursive https://github.com/deepinsight/insightface.git
```

3. Download the training set (`MS1MV2-Arcface`) and place it in *`$INSIGHTFACE_ROOT/recognition/datasets/`*. Each training dataset includes the following 6 files:

```Shell
    faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

The first three files are the training dataset while the last three files are verification sets.

4. Train deep face recognition models.
In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/recognition/ArcFace`*.

Place and edit config file:
```Shell
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```

We give some examples below. Our experiments were conducted on the Tesla P40 GPU.

(1). Train ArcFace with LResNet100E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore
```

It will output verification results of *LFW*, *CFP-FP* and *AgeDB-30* every 2000 batches. You can check all options in *config.py*.
This model can achieve *LFW 99.80+* and *MegaFace 98.3%+*.

(2). Train CosineFace with LResNet50E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore
```

(3). Train Softmax with MobileFaceNet.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network y1 --loss softmax --dataset emore
```

(4). Fine-turn the above Softmax model with Triplet loss.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network mnas05 --loss triplet --lr 0.005 --pretrained ./models/y1-softmax-emore,1
```

### Citation

If you find *ArcFace* useful in your research, please consider to cite the following related papers:

```
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
```
