简体中文 | [English](./README_en.md)

# InsightFace Paddle

## 1. 介绍
`InsightFacePaddle`是基于PaddlePaddle实现的，开源深度人脸检测、识别工具。`InsightFacePaddle`目前提供了三个预训练模型，包括用于人脸检测的 `BlazeFace`、用于人脸识别的 `ArcFace` 和 `MobileFace`。

## 2. 安装
1. 安装 PaddlePaddle

`InsightFacePaddle` 需要使用 PaddlePaddle 2.1 及以上版本，可以参考以下步骤安装。

```bash
# for GPU
pip3 install paddlepaddle-gpu

# for CPU
pip3 install paddlepaddle
```
关于安装 PaddlePaddle 的更多信息，请参考 [PaddlePaddle](https://www.paddlepaddle.org.cn/)。

2. 安装 requirements

`InsightFacePaddle` 的依赖在 `requirements.txt` 中，你可以参考以下步骤安装依赖包。

```bash
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

## 3. 快速开始

`InsightFacePaddle` 提供 `命令行（Command Line）` 和 `Python接口` 两种使用方式。

### 3.1 命令行

在命令行中使用 `InsightFacePaddle`，请参考[InsightFacePaddle](https://github.com/littletomatodonkey/insight-face-paddle)

### 3.2 Python

同样可以在 Python 中使用 `InsightFacePaddle`。首先导入 `InsightFacePaddle`，因为 `InsightFacePaddle` 使用 `logging` 控制日志输入，因此需要导入 `logging`。

```python
import insightface_paddle as face
import logging
logging.basicConfig(level=logging.INFO)
```

#### 3.2.1 获取参数信息

```python
parser = face.parser()
help_info = parser.print_help()
print(help_info)
```

#### 3.2.2 构建索引

如果使用识别功能，则在开始预测之前，必须先构建索引文件，命令和示例数据集请参考[InsightFacePaddle](https://github.com/littletomatodonkey/insight-face-paddle)。

#### 3.2.3 预测

1. 仅检测

* Image(s)

使用下图进行测试：
<div align="center">
<img src="./demo/friends/query/friends1.jpg"  width = "800" />
</div>

预测命令如下：
```python
parser = face.parser()
args = parser.parse_args()

args.det = True
args.output = "./demo/friends/output"
input_path = "./demo/friends/query/friends1.jpg"

predictor = face.InsightFace(args)
predictor.predict(input_path)
```

检测结果图位于路径 `./demo/friends/output` 下：
<div align="center">
<img src="./demo/friends/output/friends1.jpg"  width = "800" />
</div>

* NumPy
```python
import cv2

parser = face.parser()
args = parser.parse_args()

args.det = True
args.output = "./demo/friends/output"
path = "./demo/friends/query/friends1.jpg"
img = cv2.imread(path)[:, :, ::-1]

predictor = face.InsightFace(args)
predictor.predict(img)
```

* Video
```python
parser = face.parser()
args = parser.parse_args()

args.det = True
args.output = "./demo/friends/output"
input_path = "./demo/friends/query/friends.mp4"

predictor = face.InsightFace(args)
predictor.predict(input_path)
```

2. 仅识别

* Image(s)

使用下图进行测试：
<div align="center">
<img src="./demo/friends/query/Rachel.png"  width = "200" />
</div>

预测命令如下：
```python
parser = face.parser()
args = parser.parse_args()

args.rec = True
args.index = "./demo/friends/index.bin"
input_path = "./demo/friends/query/Rachel.png"

predictor = face.InsightFace(args)
predictor.predict(input_path)
```

检测结果输出在终端中：
```bash
INFO:root:File: Rachel.png, predict label(s): ['Rachel']
```

* NumPy
```python
import cv2

parser = face.parser()
args = parser.parse_args()

args.rec = True
args.index = "./demo/friends/index.bin"
path = "./demo/friends/query/Rachel.png"
img = cv2.imread(path)[:, :, ::-1]

predictor = face.InsightFace(args)
predictor.predict(img)
```

3. 检测+识别系统串联

* Image(s)

使用下图进行测试：
<div align="center">
<img src="./demo/friends/query/friends2.jpg"  width = "800" />
</div>

预测命令如下：
```python
parser = face.parser()
args = parser.parse_args()

args.det = True
args.rec = True
args.index = "./demo/friends/index.bin"
args.output = "./demo/friends/output"
input_path = "./demo/friends/query/friends2.jpg"

predictor = face.InsightFace(args)
predictor.predict(input_path)
```

检测结果图位于路径 `./demo/friends/output` 下：
<div align="center">
<img src="./demo/friends/output/friends2.jpg"  width = "800" />
</div>


* NumPy
```python
import cv2

parser = face.parser()
args = parser.parse_args()

args.det = True
args.rec = True
args.index = "./demo/friends/index.bin"
args.output = "./demo/friends/output"
path = "./demo/friends/query/friends1.jpg"
img = cv2.imread(path)[:, :, ::-1]

predictor = face.InsightFace(args)
predictor.predict(img)
```

* Video
```python
parser = face.parser()
args = parser.parse_args()

args.det = True
args.rec = True
args.index = "./demo/friends/index.bin"
args.output = "./demo/friends/output"
input_path = "./demo/friends/query/friends.mp4"

predictor = face.InsightFace(args)
predictor.predict(input_path)
```
