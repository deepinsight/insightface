# Parital FC

**The sampled version of pytorch is still being improved, 
and there is a bug which the accuracy can not reach mxnet version,
but you can use the unsampled version of pytorch first.**
 
If you want to reproduce the accuracy in the paper, it is strongly recommended to use mxnet first. 
All experiments in the paper are done by mxnet.

Pytorch 目前是还是预览版本，模型并行是没问题的，但是0.1的采样**暂时无法使用**，  
**如果要使用采样(复现论文中的精度)，强烈建议优先使用mxnet, 所有论文的实验均是mxnet完成的**。  
我们会马上修复这个bug。

Insightface 社区需要大家一起贡献才会变得更好，欢迎大家提交Pull Request.  

## TODO

- [ ] **No BUG** Sampling  
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
### MS1MV2-IJBC
```shell script
+--------------+-------+-------+--------+-------+-------+-------+
|   Methods    | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
+--------------+-------+-------+--------+-------+-------+-------+
| cosface-IJBC | 86.63 | 94.22 | 96.37  | 97.61 | 98.34 | 99.08 |
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
