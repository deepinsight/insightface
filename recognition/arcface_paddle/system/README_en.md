[简体中文](./README_ch.md) | English

# InsightFace Paddle

## 1. Introduction
`InsightFacePaddle` is an open source deep face detection and recognition toolkit, powered by PaddlePaddle. `InsightFacePaddle` provide three related pretrained models now, include `BlazeFace` for face detection, `ArcFace` and `MobileFace` for face recognition.

## 2. Prepare for the environment

Please refer to [Installation](../install_en.md) to setup environment at first.

## 3. Quick Start

You can use `InsightFacePaddle` in Python. And `InsightFacePaddle` also provide Python wheel, support two ways of use, including `Commad Line` and `Python API`. Please refer to [InsightFacePaddle](https://github.com/littletomatodonkey/insight-face-paddle) for details.

Using `InsightFacePaddle` in Python, first, you need to import `InsightFacePaddle` and `logging` because `InsightFacePaddle` using that to control log.

```python
import insightface_paddle as face
import logging
logging.basicConfig(level=logging.INFO)
```

### 3.1 Get help

```python
parser = face.parser()
help_info = parser.print_help()
print(help_info)
```

The args are as follows:
| args |  type | default | help |
| ---- | ---- | ---- | ---- |
| det_model | str | BlazeFace | The detection model. |
| rec_model | str | MobileFace | The recognition model. |
| use_gpu | bool | True | Whether use GPU to predict. Default by `True`. |
| enable_mkldnn | bool | False | Whether use MKLDNN to predict, valid only when `--use_gpu` is `False`. Default by `False`. |
| cpu_threads | int | 1 | The num of threads with CPU, valid only when `--use_gpu` is `False` and `--enable_mkldnn` is `True`. Default by `1`. |
| input | str | - | The path of video to be predicted. Or the path or directory of image file(s) to be predicted. |
| output | str | - | The directory to save prediction result. |
| det | bool | False | Whether to detect. |
| det_thresh | float | 0.8 | The threshold of detection postprocess. Default by `0.8`. |
| rec | bool | False | Whether to recognize. |
| index | str | - | The path of index file. |
| cdd_num | int | 10 | The number of candidates in the recognition retrieval. Default by `10`. |
| rec_thresh | float | 0.4 | The threshold of match in recognition, use to remove candidates with low similarity. Default by `0.4`. |
| max_batch_size | int | 1 | The maxium of batch_size to recognize. Default by `1`. |
| build_index | str | - | The path of index to be build. |
| img_dir | str | - | The img(s) dir used to build index. |
| label | str | - | The label file path used to build index. |

### 3.2 Building index

Before using recognition, you need to have the index file ready. We have provided the index file of the demo: `./demo/friends/index.bin`. If need, please refer to [InsightFacePaddle](https://github.com/littletomatodonkey/insight-face-paddle) to get the demo dataset used to build index and details about the building command.

### 3.3 Prediction

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
args.output = "./output"
input_path = "./demo/friends/query/friends1.jpg"

predictor = face.InsightFace(args)
res = predictor.predict(input_path)
print(next(res))
```

The result is under the directory `./output`. The result of demo predicted by us is as follows:
<div align="center">
<img src="./demo/friends/output/friends1.jpg"  width = "800" />
</div>

* NumPy
```python
import cv2

parser = face.parser()
args = parser.parse_args()

args.det = True
args.output = "./output"
path = "./demo/friends/query/friends1.jpg"
img = cv2.imread(path)[:, :, ::-1]

predictor = face.InsightFace(args)
res = predictor.predict(img)
print(next(res))
```

The prediction result saved as `"./output/tmp.png"`.

* Video
```python
parser = face.parser()
args = parser.parse_args()

args.det = True
args.output = "./output"
input_path = "./demo/friends/query/friends.mp4"

predictor = face.InsightFace(args)
res = predictor.predict(input_path, print_info=True)
for _ in res:
    pass
```

2. Recognition only

Using the index file (`./demo/friends/index.bin`) provided by us for demonstrating in the following examples.

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
res = predictor.predict(input_path, print_info=True)
next(res)
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
res = predictor.predict(img, print_info=True)
next(res)
```

3. Detection and recognition

Using the index file (`./demo/friends/index.bin`) provided by us for demonstrating in the following examples.

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
args.output = "./output"
input_path = "./demo/friends/query/friends2.jpg"

predictor = face.InsightFace(args)
res = predictor.predict(input_path, print_info=True)
next(res)
```

The result is under the directory `./output`. The result of demo predicted by us is as follows:
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
args.output = "./output"
path = "./demo/friends/query/friends1.jpg"
img = cv2.imread(path)[:, :, ::-1]

predictor = face.InsightFace(args)
res = predictor.predict(img, print_info=True)
next(res)
```

The prediction result saved as `"./output/tmp.png"`.

* Video
```python
parser = face.parser()
args = parser.parse_args()

args.det = True
args.rec = True
args.index = "./demo/friends/index.bin"
args.output = "./output"
input_path = "./demo/friends/query/friends.mp4"

predictor = face.InsightFace(args)
res = predictor.predict(input_path, print_info=True)
for _ in res:
    pass
```
