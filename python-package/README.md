# InsightFace Python Library

## License

The code of InsightFace Python Library is released under the MIT License. There is no limitation for both academic and commercial usage.

**The pretrained models we provided with this library are available for non-commercial research purposes only, including both auto-downloading models and manual-downloading models.**

## Install

```
pip install -U insightface
```

## Quick Example

```
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
```

This quick example will detect faces from the ``t1.jpg`` image and draw detection results on it.

## Inference Backend

For ``insightface<=0.1.5``, we use MXNet as inference backend.

(You may please download all models from [onedrive](https://1drv.ms/u/s!AswpsDO2toNKrUy0VktHTWgIQ0bn?e=UEF7C4), and put them all under `~/.insightface/models/` directory to use this old version)

Starting from insightface>=0.2, we use onnxruntime as inference backend.

(You have to install ``onnxruntime-gpu`` to enable GPU inference)

## Model Zoo

In the latest version of insightface library, we provide following model packs:


| Name      | Detection Model  | Recognition Model  | Alignment | Attributes |
| ----------------------- | -----------------   | ----- | ----- | ----- | 
| **antelopev2** | SCRFD-10GF        | ResNet100@Glint360K | 2d106 & 3d68 | Gender&Age |

**Note that these models are available for non-commercial research purposes only.**

For insightface>=0.3.3, models will be downloaded automatically once we init ``app = FaceAnalysis()`` instance.

For insightface==0.3.2, you must first download the model package by command:

```
insightface-cli model.download antelope
```
or
```
insightface-cli model.download antelopev2
```

## Use Your Own Licensed Model

You can simply create a new model directory under ``~/.insightface/models/`` and replace the pretrained models we provide with your own models. And then call ``app = FaceAnalysis(name='your_model_zoo')`` to load these models.

## Call Models

The latest insightface libary only supports onnx models. Once you have trained detection or recognition models by PyTorch, MXNet or any other frameworks, you can convert it to the onnx format and then they can be called with insightface library.

### Call Detection Models

```
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Method-1, use FaceAnalysis
app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

# Method-2, load model directly
detector = insightface.model_zoo.get_model('your_detection_model.onnx')
detector.prepare(ctx_id=0, det_size=(640, 640))

```

### Call Recognition Models

```
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

handler = insightface.model_zoo.get_model('your_recognition_model.onnx')
handler.prepare(ctx_id=0)

```


