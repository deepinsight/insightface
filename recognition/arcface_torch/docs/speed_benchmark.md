## Test Training Speed

- Test Commands

You need to use the following two commands to test the Partial FC training performance. 
The number of identites is **3 millions** (synthetic data), turn mixed precision  training on, backbone is resnet50, 
batch size is 1024.
```shell
# Model Parallel
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/3millions
# Partial FC 0.1
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/3millions_pfc
```

- GPU Memory

```
# (Model Parallel) gpustat -i
[0] Tesla V100-SXM2-32GB | 64'C,  94 % | 30338 / 32510 MB 
[1] Tesla V100-SXM2-32GB | 60'C,  99 % | 28876 / 32510 MB 
[2] Tesla V100-SXM2-32GB | 60'C,  99 % | 28872 / 32510 MB 
[3] Tesla V100-SXM2-32GB | 69'C,  99 % | 28872 / 32510 MB 
[4] Tesla V100-SXM2-32GB | 66'C,  99 % | 28888 / 32510 MB 
[5] Tesla V100-SXM2-32GB | 60'C,  99 % | 28932 / 32510 MB 
[6] Tesla V100-SXM2-32GB | 68'C, 100 % | 28916 / 32510 MB 
[7] Tesla V100-SXM2-32GB | 65'C,  99 % | 28860 / 32510 MB 

# (Partial FC 0.1) gpustat -i
[0] Tesla V100-SXM2-32GB | 60'C,  95 % | 10488 / 32510 MB                                                                                                                                          │·······················
[1] Tesla V100-SXM2-32GB | 60'C,  97 % | 10344 / 32510 MB                                                                                                                                          │·······················
[2] Tesla V100-SXM2-32GB | 61'C,  95 % | 10340 / 32510 MB                                                                                                                                          │·······················
[3] Tesla V100-SXM2-32GB | 66'C,  95 % | 10340 / 32510 MB                                                                                                                                          │·······················
[4] Tesla V100-SXM2-32GB | 65'C,  94 % | 10356 / 32510 MB                                                                                                                                          │·······················
[5] Tesla V100-SXM2-32GB | 61'C,  95 % | 10400 / 32510 MB                                                                                                                                          │·······················
[6] Tesla V100-SXM2-32GB | 68'C,  96 % | 10384 / 32510 MB                                                                                                                                          │·······················
[7] Tesla V100-SXM2-32GB | 64'C,  95 % | 10328 / 32510 MB                                                                                                                                        │·······················
```

- Training Speed

```python
# (Model Parallel) trainging.log
Training: Speed 2271.33 samples/sec   Loss 1.1624   LearningRate 0.2000   Epoch: 0   Global Step: 100 
Training: Speed 2269.94 samples/sec   Loss 0.0000   LearningRate 0.2000   Epoch: 0   Global Step: 150 
Training: Speed 2272.67 samples/sec   Loss 0.0000   LearningRate 0.2000   Epoch: 0   Global Step: 200 
Training: Speed 2266.55 samples/sec   Loss 0.0000   LearningRate 0.2000   Epoch: 0   Global Step: 250 
Training: Speed 2272.54 samples/sec   Loss 0.0000   LearningRate 0.2000   Epoch: 0   Global Step: 300 

# (Partial FC 0.1) trainging.log
Training: Speed 5299.56 samples/sec   Loss 1.0965   LearningRate 0.2000   Epoch: 0   Global Step: 100  
Training: Speed 5296.37 samples/sec   Loss 0.0000   LearningRate 0.2000   Epoch: 0   Global Step: 150  
Training: Speed 5304.37 samples/sec   Loss 0.0000   LearningRate 0.2000   Epoch: 0   Global Step: 200  
Training: Speed 5274.43 samples/sec   Loss 0.0000   LearningRate 0.2000   Epoch: 0   Global Step: 250  
Training: Speed 5300.10 samples/sec   Loss 0.0000   LearningRate 0.2000   Epoch: 0   Global Step: 300   
```

In this test case, Partial FC 0.1 only use1 1/3 of the GPU memory of the model parallel, 
and the training speed is 2.5 times faster than the model parallel.