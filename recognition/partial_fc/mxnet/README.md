## Train
#### Requirements
python==3.6  
cuda==10.1    
cudnn==765    
mxnet-cu101==1.6.0.post0  
pip install easydict mxboard opencv-python tqdm    
[nccl](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)  
[openmpi](mxnet/setup-utils/install-mpi.sh)==4.0.0  
[horovod](mxnet/setup-utils/install-horovod.sh)==0.19.2  

#### Failures due to SSH issues
The host where horovodrun is executed must be able to SSH to all other hosts without any prompts.

#### Run with horovodrun
Typically one GPU will be allocated per process, so if a server has 8 GPUs, you will run 8 processes. 
In horovodrun, the number of processes is specified with the -np flag.

To run on a machine with 8 GPUs:
```shell script
horovodrun -np 8 -H localhost:8 bash config.sh
```

To run on two machine with 16 GPUs:
```shell script
horovodrun -np 16 -H ip1:8,ip2:8 bash config.sh
```

#### Run with mpi
```shell script
bash run.sh
```


## Troubleshooting

### Mxnet Version!
Some versions of horovod and mxnet will have bugs.   
It is recommended to try version **1.5 or 1.6**.

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



