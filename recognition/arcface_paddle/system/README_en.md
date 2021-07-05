[简体中文](./README_ch.md) | English

# InsightFace Paddle

## 1. Introduction
`InsightFacePaddle` is an open source deep face detection and recognition toolkit, powered by PaddlePaddle. `InsightFacePaddle` provide three related pretrained models now, include `BlazeFace` for face detection, `ArcFace` and `MobileFace` for face recognition.

## 2. Installation
1. Install PaddlePaddle

PaddlePaddle 2.1 or later is required for `InsightFacePaddle`. You can use the following steps to install PaddlePaddle.

```bash
# for GPU
pip3 install paddlepaddle-gpu

# for CPU
pip3 install paddlepaddle
```
For more details about installation. please refer to [PaddlePaddle](https://www.paddlepaddle.org.cn/).

2. Install requirements

`InsightFacePaddle` dependencies are listed in `requirements.txt`, you can use the following command to install the dependencies.

```bash
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

## 3. Quick Start

`InsightFacePaddle` support two ways of use, including `Commad Line` and `Python API`.

### 3.1 Command Line

Please refer to the [InsightFacePaddle](https://github.com/littletomatodonkey/insight-face-paddle) for details about using `InsightFacePaddle` in Command Line.

### 3.2 Python

You can use `InsightFacePaddle` in Python. First, import `InsightFacePaddle` and `logging` because `InsightFacePaddle` using that to control log.

```python
import insightface_paddle as face
import logging
logging.basicConfig(level=logging.INFO)
```

#### 3.2.1 Get help

```python
parser = face.parser()
help_info = parser.print_help()
print(help_info)
```

#### 3.2.2 Building index

If use recognition, before start predicting, you have to build the index. Please refer to [InsightFacePaddle](https://github.com/littletomatodonkey/insight-face-paddle) for details about the command and demo dataset.

#### 3.2.3 Prediction

1. Detection only

* Image(s)

Use the image below to predict:
<div align="center">
<img src="./demo/friends/query/friends1.jpg"  width = "800" />
</div>

The prediction command:
```python
parser = face.parser()
args = parser.parse_args()

args.det = True
args.output = "./demo/friends/output"
input_path = "./demo/friends/query/friends1.jpg"

predictor = face.InsightFace(args)
predictor.predict(input_path)
```

The result is under the directory `./demo/friends/output`:
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

The prediction result saved as `"./demo/friends/output/tmp.png"`.

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

2. Recognition only

* Image(s)

Use the image below to predict:
<div align="center">
<img src="./demo/friends/query/Rachel.png"  width = "200" />
</div>

The prediction command:
```python
parser = face.parser()
args = parser.parse_args()

args.rec = True
args.index = "./demo/friends/index.bin"
input_path = "./demo/friends/query/Rachel.png"

predictor = face.InsightFace(args)
predictor.predict(input_path)
```

The result is output in the terminal:
```bash
INFO:root:File: Rachel., predict label(s): ['Rachel']
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

3. Detection and recognition

* Image(s)

Use the image below to predict:
<div align="center">
<img src="./demo/friends/query/friends2.jpg"  width = "800" />
</div>

The prediction command:
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

The result is under the directory `./demo/friends/output`:
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

The prediction result saved as `"./demo/friends/output/tmp.png"`.

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
