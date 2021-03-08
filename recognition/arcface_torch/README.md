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
--model-prefix ms1mv3_r50_arcface/backbone.pth \
--image-path IJB_release/IJBC \
--result-dir ms1mv3_r50_arcface \
--batch-size 128 \
--job ms1mv3_r50_arcface \
--target IJBC \
--network iresnet50
```
More details see [eval.md](docs/eval.md) in docs.


## Model Zoo
### MS1MV3
|   Datasets          | backbone | IJBC(1e-05) | IJBC(1e-04) |agedb30|cfp_fp|lfw  | 
| :---:               | :---     | :---        | :---        |:---   |:---  |:--- |  
| MS1MV3-Arcface      | r18      |             |             |       |      |     | 
| MS1MV3-Arcface      | r34      |             |             |       |      |     | 
| MS1MV3-Arcface      | r50      |   94.79     |  96.43      |98.28  |98.89 |99.85| 
| MS1MV3-Arcface      | r100     |   95.22     |  96.87      |98.45  |99.19 |99.85| 

### Glint360k
|   Datasets          | backbone | IJBC(1e-05) | IJBC(1e-04) |agedb30|cfp_fp|lfw  | 
| :---:               | :---     | :---        | :---        |:---   |:---  |:--- |
| Glint360k-Cosface   | r100     | -           | -           |-      |-     |-    |


More details see [eval.md](docs/modelzoo.md) in docs.