## 如何安装
 
#### 1. python依赖  
使用以下命令
```shell script
pip install easydict mxboard opencv-python tqdm     
```

#### 2. nccl  
nccl可以不用装，但是装上速度更快，nccl安装需要对应cuda版本，安装方法参考下边链接:  
[**NCCL**](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)  

#### 3. openmpi  
openmpi必须安装，必须采纳我的脚本编译源码安装：  
[**OpenMPI**](mxnet/setup-utils/install-mpi.sh)    

#### 4. horovod, mxnet
有些版本的mxnet的horovod无法安装，参考下方表格，强烈建议使用**mxnet==1.6.0**和**cuda==10.1**

| mxnet |horovod  |  cuda        | 
| :---: | :---    |  :---:       | 
| 1.4.0 | x       |  x           | 
| 1.5.0 | 可以安装 | cuda10.0     | 
| 1.5.1 | x       | x            | 
| 1.6.0 | 可以安装 | cuda10.1     | 
| 1.7.0 | x       | x            | 

horovod 安装方法如下:  
[**Horovod**](mxnet/setup-utils/install-horovod.sh)

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


#### 5. ssh无密登录

使用多机分布式训练的时候，每台机器都需要设置无密登录，包括自己与自己，无密码登录具体可见：  
这里推荐一个简单的命令：  
```shell script
ssh-copy-id user@ip
```


## 如何运行  
`horovod`底层调用的还是`mpi`，mpi的概念是，你有多少块GPU，就要启动多少个进程，有两种方法启动训练，使用`horovodrun`或者`mpirun`。  
#### 1. 使用 horovodrun 运行  

运行8卡(单机)：
```shell script
horovodrun -np 8 -H localhost:8 bash config.sh
```

运行16卡(两台机器)
```shell script
horovodrun -np 16 -H ip1:8,ip2:8 bash config.sh
```

#### 2. 使用 mpirun 运行  

```shell script
bash run.sh
```




#### Failures due to SSH issues
The host where horovodrun is executed must be able to SSH to all other hosts without any prompts.




## Troubleshooting

### Horovod installed successfully?  

Run `horovodrun --check` to check the installation of horovod.


### Mxnet Version!
Some versions of mxnet with horovod have bug.   
It is recommended to try version **1.5 or 1.6**.

**The community has found that mxnet1.5.1 cannot install horovod.**

### Check CUDA version!
```shell script
# Make sure your cuda version is same as mxnet, such as mxnet-cu101 (CUDA 10.1)

/usr/local/cuda/bin/nvcc -V
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2019 NVIDIA Corporation
# Built on Wed_Apr_24_19:10:27_PDT_2019
# Cuda compilation tools, release 10.1, V10.1.168
```

### Block IO
You can turn on the debug mode to check whether your slow training speed is the cause of IO.

### Training Speed.
If you find that your training speed is the io bottleneck, you can mount dataset to RAM, 
using the following command.
```shell script
# If your RAM has 256G
sudo mkdir /train_tmp
mount -t tmpfs -o size=140G  tmpfs /train_tmp
```



