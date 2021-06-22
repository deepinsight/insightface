# Installation

---

## Introduction

This document introduces how to install ArcFace-Paddle and its requirements.

## Install PaddlePaddle

Python 3.x, CUDA 10.0, CUDNN7.6.4 nccl2.1.2 and later version are required at first, For now, ArcFace-Paddle only support training on the GPU device. Please follow the instructions in the [Installation](http://www.paddlepaddle.org.cn/install/quick) if the PaddlePaddle on the device is lower than 2.0.0.


### Install PaddlePaddle using pip

If you want to use PaddlePaddle on GPU, you can use the following command to install PaddlePaddle.

```bash
pip3 install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
```

If you want to use PaddlePaddle on CPU, you can use the following command to install PaddlePaddle.

```bash
pip3 install paddlepaddle --upgrade -i https://mirror.baidu.com/pypi/simple
```

### Install PaddlePaddle from source code

You can also compile PaddlePaddle from source code, please refer to [Installation](http://www.paddlepaddle.org.cn/install/quick).

Verify Installation

```python
import paddle
paddle.utils.run_check()
```

Check PaddlePaddle version：

```bash
python3 -c "import paddle; print(paddle.__version__)"
```

Note:

- Make sure the compiled version is later than PaddlePaddle2.0.
- Indicate **WITH_DISTRIBUTE=ON** when compiling, Please refer to [Instruction](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3) for more details.
- When running in docker, in order to ensure that the container has enough shared memory for data read acceleration of Paddle, please set the parameter `--shm_size=8g` at creating a docker container, if conditions permit, you can set it to a larger value.


## Install requirements

```bash
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

If the install process of visualdl failed, you can try the following commands.

```bash
pip3 install --upgrade visualdl -i https://mirror.baidu.com/pypi/simple
```

What's more, visualdl is just supported in python3, so python3 is needed if you want to use visualdl.
