## pytorch 训练样例

[训练样例地址]()

### 下载数据集

* 下载 MS1MV3 [Link](https://github.com/deepinsight/insightface/tree/master/challenges/iccv19-lfr)
* 下载 Glint360K [Link](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download)

### 服务器提交地址

http://iccv21-mfr.com/

### 安装依赖

1. 安装 pytorch 1.7.1

假设你已经安装好了GPU驱动和CUDA，根据你的CUDA版本，来选择你要安装的pytorch命令。  
查看CUDA版本的命令为: `/usr/local/cuda/bin/nvcc -V`。

Linux and Windows  
```shell
# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

```

你也可以安装pytorch的其他版本，例如1.6.0或者更高的版本。

2. 安装其他依赖

```shell
pip install -r requirement.txt
```

### 运行
根据你的服务器，选择你要运行的命令。

* 一台服务器，四张GPU运行

```shell
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py
```

* 一台服务器，八张GPU运行

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py
```

* 多台服务器，每台服务器8张GPU

1. 节点0
```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=1234 train.py
```

2. 节点1
```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=1234 train.py
```


### 提交

1. 提交onnx模型

竞赛要求模型转换为`onnx`模型提交，arcface_torch工程在保存模型时，会自动转换成为onnx，其地址为`${cfg.output}/backbone.onnx`。

模型checkpoint介绍：
```shell
├── backbone.onnx                        # 需要提交的模型
├── backbone.pth                         # pytorch 保存的模型
├── rank_0_softmax_weight_mom.pt         # 模型并行原因，每张卡保存softmax独有的参数
├── rank_0_softmax_weight.pt
├── rank_1_softmax_weight_mom.pt
├── rank_1_softmax_weight.pt
├── ... ...
└── training.log                          # 训练日志
```

2. 检查onnx模型是否规范

提交模型前检查一下提交的模型是否规范，并测试模型的推理时间  


测试命令：
```shell
python onnx_helper_sample.py --model_root ms1mv3_arcface_r50/
```

也可以先测试一下onnx模型在公开测试集IJBC上的性能：
https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/onnx_ijbc.py

测试命令：
```shell
CUDA_VISIBLE_DEVICES=0 python onnx_ijbc.py --model-root ms1mv3_arcface_r50 --image-path IJB_release/IJBC --result-dir ms1mv3_arcface_r50
```

3. 模型大小参考

推理时间是在`Tesla V100 GPU`中测试, 其中 onnxruntime-gpu==1.6。

| 模型名称      | 大小/MB     | 推理时间/ms      |  
| -------      | ----------  | -----------   |
| R50          | 166         | 4.262         | 
| R100         | 248         | 7.031         |  
| R200         | 476         | 13.48         | 

### 提示与技巧

1. 训练加速-混合精度训练

当时使用图灵架构的GPU时候，强烈建议开启混合精度训练模型，在`config.py`中，将`config.fp16`设置为True，可以节省大量显存和提升训练速度，例如：

训练设置：
MS1MV3(SSD) + 4*V100 + R100 + BatchSize 4*128

- 开启混合精度训练前
```python3
# training log
Training: 2021-05-12 00:00:42,110-Speed 884.42 samples/sec   Loss 47.2532   Epoch: 0   Global Step: 100
Training: 2021-05-12 00:01:10,979-Speed 886.77 samples/sec   Loss 47.3550   Epoch: 0   Global Step: 150
Training: 2021-05-12 00:01:43,936-Speed 776.80 samples/sec   Loss 47.0214   Epoch: 0   Global Step: 200
Training: 2021-05-12 00:02:16,064-Speed 796.83 samples/sec   Loss 46.7781   Epoch: 0   Global Step: 250
Training: 2021-05-12 00:02:45,018-Speed 884.18 samples/sec   Loss 46.3187   Epoch: 0   Global Step: 300
# gpustat -i
[0] Tesla V100-SXM2-32GB | 67 C,  99 % | 17844 / 32510 MB 
[1] Tesla V100-SXM2-32GB | 64 C,  98 % | 17844 / 32510 MB 
[2] Tesla V100-SXM2-32GB | 65 C,  93 % | 17916 / 32510 MB 
[3] Tesla V100-SXM2-32GB | 72 C,  82 % | 17910 / 32510 MB 
```

- 开启混合精度训练后

```python3
# training log
Training: 2021-05-12 00:04:27,869-Speed 1604.59 samples/sec   Loss 47.6050   Epoch: 0   Global Step: 100
Training: 2021-05-12 00:04:43,681-Speed 1619.08 samples/sec   Loss 47.5865   Epoch: 0   Global Step: 150
Training: 2021-05-12 00:04:59,460-Speed 1622.39 samples/sec   Loss 47.2380   Epoch: 0   Global Step: 200
Training: 2021-05-12 00:05:15,271-Speed 1619.25 samples/sec   Loss 46.9030   Epoch: 0   Global Step: 250
Training: 2021-05-12 00:05:31,065-Speed 1620.86 samples/sec   Loss 46.4425   Epoch: 0   Global Step: 300
# gpustat -i
[0] Tesla V100-SXM2-32GB | 64 C,  96 % | 10664 / 32510 M  
[1] Tesla V100-SXM2-32GB | 63 C,  96 % | 10630 / 32510 MB 
[2] Tesla V100-SXM2-32GB | 63 C,  79 % | 10736 / 32510 MB 
[3] Tesla V100-SXM2-32GB | 70 C,  86 % | 10736 / 32510 MB
```

2. 训练加速-将数据挂载到内存盘来提升训练速度  
使用如下的命令：
```shell
# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=40G  tmpfs /train_tmp
```

让后将训练集拷贝到目录`/train_tmp`下，然后开始训练。
