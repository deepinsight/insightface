# Parital FC

Pytorch is currently still a preview version. There is a 5 thousandth difference between the sampling of 0.1 and the paper.
**If you want to reproduce the accuracy in the paper, it is strongly recommended to use mxnet first. ** 
All experiments in the paper are done by mxnet.

Pytorch 目前是还是预览版本，目前0.1的采样和论文中有点微小的差距，  
**如果要复现论文中的精度，强烈建议优先使用mxnet, 所有论文的实验均是mxnet完成的**。

Insightface 社区需要大家一起贡献才会变得更好，欢迎大家提交Pull Request.  


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
