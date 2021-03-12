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

## Speed Benchmark
![Image text](https://github.com/nttstar/insightface-resources/blob/master/images/arcface_speed.png)

ArcFace_torch can train large-scale face recognition training set efficiently and quickly.  
When the number of classes in training sets is greater than 300K and the training is sufficient, 
partial fc sampling strategy will get same accuracy with several times faster training performance and smaller GPU memory.

1. Training speed of different parallel methods (samples/second), Tesla V100 32GB * 8. (Larger is better)

| Method                 | Bs128-R100-2 Million Identities | Bs128-R50-4 Million Identities | Bs64-R50-8 Million Identities |
| :---                   |    :---                          | :---                            | :---                     |
| Data Parallel          |    1                             | 1                               | 1                        |
| Model Parallel         |    1362                          | 1600                            | 482                      |
| Fp16 + Model Parallel  |    2006                          | 2165                            | 767                      | 
| Fp16 + Partial Fc 0.1  |    3247                          | 4385                            | 3001                     | 

2. GPU memory cost of different parallel methods (GB per GPU), Tesla V100 32GB * 8. (Smaller is better)

| Method                 | Bs128-R100-2 Million Identities   | Bs128-R50-4 Million Identities   | Bs64-R50-8 Million Identities |
| :---                   |    :---                           | :---                             | :---                     |
| Data Parallel          |    OOM                            | OOM                              | OOM                      |
| Model Parallel         |    27.3                           | 30.3                             | 32.1                      |
| Fp16 + Model Parallel  |    20.3                           | 26.6                             | 32.1                      | 
| Fp16 + Partial Fc 0.1  |    11.9                           | 10.8                             | 11.1                      | 


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
