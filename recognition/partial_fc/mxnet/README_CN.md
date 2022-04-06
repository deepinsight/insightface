## 目录
## Contents
[Partial FC](https://arxiv.org/abs/2203.15565)
- [如何安装](#如何安装)
- [如何运行](#如何运行)
- [错误排查](#错误排查)


## 如何安装
 
### 1. python依赖  
使用以下命令
```shell script
pip install easydict mxboard opencv-python tqdm     
```

### 2. 安装nccl  
nccl可以不用装，但是装上速度更快，nccl安装需要对应cuda版本，安装方法参考下边链接:  
[**NCCL**](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)  

### 3. 安装openmpi  
openmpi必须安装，必须采纳我的脚本编译源码安装：  
[**OpenMPI**](setup-utils/install-mpi.sh)    

### 4. 安装horovod, mxnet
有些版本的mxnet的horovod无法安装，参考下方表格，强烈建议使用**mxnet==1.6.0**和**cuda==10.1**

| mxnet |horovod  |  cuda        | 
| :---: | :---    |  :---:       | 
| 1.4.0 | x       |  x           | 
| 1.5.0 | 可以安装 | cuda10.0     | 
| 1.5.1 | x       | x            | 
| 1.6.0.post0 | 可以安装 | cuda10.1     | 
| 1.7.0 | x       | x            | 

horovod 安装方法如下:  
[**Horovod**](setup-utils/install-horovod.sh)

horovod 安装完成后使用下面的命令检查horovod是否安装成功，(nccl有没有都可以，有nccl会更快)：
```shell script
# Horovod v0.19.2:
# Available Frameworks:
#     [ ] TensorFlow
#     [ ] PyTorch
#     [X] MXNet
# 
# Available Controllers:
#     [X] MPI
#     [X] Gloo
# 
# Available Tensor Operations:
#     [X] NCCL
#     [ ] DDL
#     [ ] CCL
#     [X] MPI
#     [X] Gloo
```


### 5. ssh无密登录

使用多机分布式训练的时候，每台机器都需要设置无密登录，包括自己与自己，无密码登录具体可见：  
这里推荐一个简单的命令：  
```shell script
ssh-copy-id user@ip
```

## 如何运行  
`horovod`底层调用的还是`mpi`，mpi的概念是，你有多少块GPU，就要启动多少个进程，有两种方法启动训练，使用`horovodrun`或者`mpirun`。  
### 1. 使用 horovodrun 运行  

运行8卡(单机)：
```shell script
horovodrun -np 8 -H localhost:8 bash config.sh
```

运行16卡(两台机器)
```shell script
horovodrun -np 16 -H ip1:8,ip2:8 bash config.sh
```

### 2. 使用 mpirun 运行  

```shell script
bash run.sh
```

## 错误排查

QQ群：711302608  

### 检查Horovod是否安装成功？

运行这个命令 `horovodrun --check` 来检查horovod是否安装成功。

### 检查你的CUDA版本是否与mxnet匹配，比如mxnet-cu101需要的cuda版本为CUDA10.1  

```shell script
# Make sure your cuda version is same as mxnet, such as mxnet-cu101 (CUDA 10.1)

/usr/local/cuda/bin/nvcc -V
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2019 NVIDIA Corporation
# Built on Wed_Apr_24_19:10:27_PDT_2019
# Cuda compilation tools, release 10.1, V10.1.168
```

### 屏蔽IO对训练速度的影响？  

可以在`config.py`中开启debug模式，来屏蔽IO，看看是否是IO对性能的影响。

### 将数据挂载到内存盘来提高训练速度。

如果你发现你训练速度的瓶颈是IO的话，你可以把数据挂载到内存盘来提高训练的速度，挂载的命令如下：  
需要注意的是，你的RAM必须足够的大。

```shell script
# If your RAM has 256G
sudo mkdir /train_tmp
mount -t tmpfs -o size=140G  tmpfs /train_tmp
```



