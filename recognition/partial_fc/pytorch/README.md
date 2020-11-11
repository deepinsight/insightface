# Parital FC

## TODO

- [x] **No BUG** Sampling  
- [ ] Mixed precision training  
- [ ] Pipeline Parallel  
- [ ] Checkpoint  
- [ ] Docker  
- [ ] A Wonderful Documents  

## How to run
cuda=10.1  
pytorch==1.6.0  
pip install -r requirement.txt  

```shell
bash run.sh
```
使用 `bash run.sh` 这个命令运行。

## Results
There is a loss of accuracy in MS1MV2 with sampling, 
but it has the same good accuracy in Glint360k.  


### IJBC-Glint360K (mxnet)
```shell script
+--------------+-------+-------+--------+-------+-------+-------+
|   Methods    | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
+--------------+-------+-------+--------+-------+-------+-------+
| IJBC-1.0     | 91.29 | 95.92 | 97.30  | 98.13 | 98.78 | 99.28 |
+--------------+-------+-------+--------+-------+-------+-------+
| IJBC-0.1     | 91.25 | 95.89 | 97.27  | 98.12 | 98.77 | 99.28 |
+--------------+-------+-------+--------+-------+-------+-------+
```

### IJBC-Glint360K (pytorch)
```shell script
+--------------+-------+-------+--------+-------+-------+-------+
|   Methods    | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
+--------------+-------+-------+--------+-------+-------+-------+
| IJBC-1.0     |       |       |        |       |       |       |
+--------------+-------+-------+--------+-------+-------+-------+
| IJBC-0.1     |       |       |        |       |       |       |
+--------------+-------+-------+--------+-------+-------+-------+
```


### IJBC-MS1MV2 (pytorch)
```shell script
+--------------+-------+-------+--------+-------+-------+-------+
|   Methods    | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
+--------------+-------+-------+--------+-------+-------+-------+
| IJBC-1.0     | 86.63 | 94.22 | 96.27  | 97.61 | 98.34 | 99.08 |
+--------------+-------+-------+--------+-------+-------+-------+
| IJBC-0.1     | 76.76 | 92.34 | 96.24  | 97.61 | 98.51 | 99.16 |
+--------------+-------+-------+--------+-------+-------+-------+
```




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
