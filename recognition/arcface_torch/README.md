# Arcface Pytorch (Distributed Version of ArcFace)


## Contents

## Set Up
```shell
torch >= 1.6.0
```

## Train on a single node 
```shell
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python -m torch.distributed.launch \ 
--nproc_per_node=8 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" \
--master_port=1234 train.py
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
```

## Train on multi-node
```shell
pass
```

## Evaluation
```shell
# model-prefix       your model path
# image-path         your IJBC path
# result-dir         your result path
# network            your backbone
CUDA_VISIBLE_DEVICES=0,1 python eval_ijbc.py \
--model-prefix ms1mv3_arcface_r50/backbone.pth \
--image-path IJB_release/IJBC \
--result-dir ms1mv3_arcface_r50 \
--batch-size 128 \
--job ms1mv3_arcface_r50 \
--target IJBC \
--network iresnet50
```
More details see [eval.md](docs/eval.md) in docs.


## Model Zoo
### MS1MV3
|   Datasets          | backbone | IJBC(1e-05) | IJBC(1e-04) |agedb30|cfp_fp|lfw  | 
| :---:               | :---     | :---        | :---        |:---   |:---  |:--- |  
| MS1MV3-Arcface      | r18      |   92.08     |  94.68      |97.65  |97.63 |99.73|
| MS1MV3-Arcface      | r34      |             |             |       |      |     | 
| MS1MV3-Arcface      | r50      |   94.79     |  96.43      |98.28  |98.89 |99.85| 
| MS1MV3-Arcface      | r50-amp  |   94.72     |  96.41      |98.30  |99.06 |99.85| 
| MS1MV3-Arcface      | r100     |   95.22     |  96.87      |98.45  |99.19 |99.85| 

### Glint360k
|   Datasets          | backbone | IJBC(1e-05) | IJBC(1e-04) |agedb30|cfp_fp|lfw  | 
| :---:               | :---     | :---        | :---        |:---   |:---  |:--- |
| Glint360k-Cosface   | r100     |  96.19      | 97.39       |98.52  |99.26 |99.83|

More details see [eval.md](docs/modelzoo.md) in docs.



## Citation
```
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
@inproceedings{an2020partical_fc,
  title={Partial FC: Training 10 Million Identities on a Single Machine},
  author={An, Xiang and Zhu, Xuhan and Xiao, Yang and Wu, Lan and Zhang, Ming and Gao, Yuan and Qin, Bin and
  Zhang, Debing and Fu Ying},
  booktitle={Arxiv 2010.05222},
  year={2020}
}
```