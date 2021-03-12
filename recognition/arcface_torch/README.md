# Arcface Pytorch (Distributed Version of ArcFace)


## Contents

## Set Up
```shell
torch >= 1.6.0
```

## Train on a single node 
If you want to use 8 GPU to train, you should set `--nproc_per_node=8` and set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 `  
If you want to use 4 GPU to train, you should set `--nproc_per_node=4` and set `CUDA_VISIBLE_DEVICES=0,1,2,3`  
If you want to use 1 GPU to train, you should set `--nproc_per_node=1` ...  

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

The models are available for non-commercial research purposes only.

All Model Can be found in here.  
[Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g):   e8pw  

### MS1MV3
|   Datasets          |    log     | backbone    | IJBC(1e-05) | IJBC(1e-04) |agedb30|cfp_fp|lfw  | 
| :---:               |    :---    | :---        | :---        | :---        |:---   |:---  |:--- |  
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r18/training.log)             | r18                 | 92.08 | 94.68 | 97.65 | 97.63 | 99.73 |
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r34/training.log)             | r34                 | 94.13 | 95.98 | 98.05 | 98.60 | 99.80 | 
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r50/training.log)             | r50                 | 94.79 | 96.43 | 98.28 | 98.89 | 99.85 | 
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r50_fp16/training.log)        | r50-fp16            | 94.72 | 96.41 | 98.30 | 99.06 | 99.85 | 
| MS1MV3-Arcface      |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_arcface_r100/training.log)            | r100                | 95.22 | 96.87 | 98.45 | 99.19 | 99.85 | 
   
### Glint360k
|   Datasets          | log   |backbone               | IJBC(1e-05) | IJBC(1e-04) |agedb30|cfp_fp|lfw  | 
| :---:               | :---  |:---                   | :---        | :---        |:---   |:---  |:--- |
| Glint360k-Cosface   |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r100/training.log)         |r100                 | 96.19 | 97.39 | 98.52 | 99.26 | 99.83 |
| Glint360k-Cosface   |[log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_cosface_r100_fp16_0.1/training.log)|r100-fp16-sample-0.1 | 95.95 | 97.35 | 98.57 | 99.30 | 99.85 |

More details see [eval.md](docs/modelzoo.md) in docs.