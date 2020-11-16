# Parital FC

## TODO

- [x] **No BUG** Sampling  
- [ ] Pytorch Experiments (Glint360k, 1.0/0.1)   
- [ ] Mixed precision training  
- [ ] Pipeline Parallel  
- [ ] Checkpoint  
- [ ] Docker  
- [ ] A Wonderful Documents  

## Results
We employ ResNet100 as the backbone.

### 1. IJB-C results

|   Datasets   | 1e-05 | 1e-04 | 1e-03 | 1e-02 | 1e-01 |
| :---:        | :---  | :---  | :---  | :---  | :---  | 
| Glint360K    | 95.92 | 97.30 | 98.13 | 98.78 | 99.28 |
| MS1MV2       | 94.22 | 96.27 | 97.61 | 98.34 | 99.08 |

### 2. IFRT results

TODO

## Training Speed Benchmark
### 1. Train MS1MV2
We employ **ResNet100** as the backbone.   
|   GPU                   | FP16  | BatchSize / it | GPUs / it  | Throughput img / sec | Time / hours |
| :---                    | :---  |   :---         | :---       | :---                 | :---         | 
| Tesla V100-SXM2-32GB    | False |   64           | 8          | 1658                 | 15           |
| Tesla V100-SXM2-32GB    | False |   128          | 8          | 1800                 | 14           |    
| Tesla V100-SXM2-32GB    | True  |   64           | 8          | 2198                 | 12           |
| Tesla V100-SXM2-32GB    | True  |   128          | 8          | 3337                 | 7            |     
| RTX2080Ti               | False |                | 8          | 1200                 |              | 
| RTX2080Ti               |       |                | 8          |                      |              | 

### 2. Train millions classes
TODO

## How to run
cuda=10.1  
pytorch==1.6.0  
pip install -r requirement.txt  

```shell
bash run.sh
```
使用 `bash run.sh` 这个命令运行。


## Citation
If you find Partial-FC or Glint360K useful in your research, please consider to cite the following related paper: 

[Partial FC](https://arxiv.org/abs/2010.05222)
```
@inproceedings{an2020partical_fc,
  title={Partial FC: Training 10 Million Identities on a Single Machine},
  author={An, Xiang and Zhu, Xuhan and Xiao, Yang and Wu, Lan and Zhang, Ming and Gao, Yuan and Qin, Bin and
  Zhang, Debing and Fu Ying},
  booktitle={Arxiv 2010.05222},
  year={2020}
}
```
