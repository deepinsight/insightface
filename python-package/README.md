# InsightFace Python Library

## License

The code of InsightFace Python Library is released under the MIT License. There is no limitation for both academic and commercial usage.

**The pretrained models we provided with this library are available for non-commercial research purposes only, including both auto-downloading models and manual-downloading models.**

## Install

### Install Inference Backend

For ``insightface<=0.1.5``, we use MXNet as inference backend.

Starting from insightface>=0.2, we use onnxruntime as inference backend.

You have to install ``onnxruntime-gpu`` manually to enable GPU inference, install ``onnxruntime-cann`` manually to enable Ascend NPU inference, or install ``onnxruntime`` to use CPU only inference.

### Install InsightFace Evaluation Studio GUI

InsightFace 1.0.1 includes a local desktop GUI:

```
pip install "insightface[gui]"
insightface-gui
```

Development install:

```
cd python-package
pip install -e ".[gui]"
insightface-gui
```

Equivalent launch commands:

```
insightface-eval-studio
insightface-desktop
python -m insightface.gui
```

The GUI is called **InsightFace Evaluation Studio**. It provides local 1:1 face
compare, People Library management, 1:N face search, multi-face photo
recognition, batch folder processing, album people clustering, enterprise
evaluation reports, and a face swap entry point. User images, videos,
embeddings, databases, and reports are stored locally by default under
``~/.insightface/gui`` and are not uploaded automatically.
Image and video previews are clickable upload targets: click
``Click to upload or drag a file here`` or drop a file onto the preview. The
preview changes color on hover and during drag-over. Loaded previews show a
small delete button and can be replaced by dragging in another file.

The desktop app uses mode-based navigation. Choose **Face Recognition**,
**Album Management**, **Face Swap**, or **Enterprise Evaluation** from the
persistent **Workflows** rail on the left side of the window. Face Recognition
is a single **Query & Gallery** workspace: upload
one query image and one gallery image for 1:1 compare, or upload multiple
gallery images / a folder for 1:N gallery search. Album Management uses a
single **Album** workspace for adding one or more folders, refreshing new
images, DBSCAN clustering with a default cosine similarity threshold of `0.48`,
and reviewing original photo thumbnails. Album directories and clustering results are saved
locally for the next launch. Enterprise Evaluation is a single workspace for
local 1:1 and 1:N identity-folder evaluation, Auto Split, metrics, and PDF
report export. Enterprise datasets must pass validation before evaluation; the
validator checks folder layout, gallery/probe rules, and the selected
multi-face handling policy.
Global utilities are available from the top bar and **Tools** menu:
**Settings**, **Models**, and **License**. **Settings** controls the UI theme
and language. Language defaults to the operating system when it is supported,
otherwise English. Supported GUI languages are English, Chinese, Japanese,
Korean, Spanish, French, German, Portuguese, and Russian. Available themes
include System, Precision Light, Studio Dark, Graphite Pro, Azure Lab, Emerald
Focus, and Crimson Audit. Workspace paths are chosen on first launch and are
not changed from the settings dialog.

Models are not downloaded automatically by the GUI. Open **Models > Downloads**,
click **Refresh Download URLs** to read the latest GitHub Releases
asset URLs, then explicitly download the selected package. Downloaded zip files
are cached under ``~/.insightface/gui/cache/models`` and extracted under
``~/.insightface/models/<model_name>/``.
The Downloads tab also lists GFPGANv1.4 as a third-party face restoration
model. After it is downloaded, enable **GFPGAN post-processing** in
**Models > Runtime** to run 512x512 GFPGAN restoration after face swap.
Detection size defaults to **Auto**, which runs joint 128x128 and 640x640
detection. Face swap models are selected in **Models > Runtime** from already
downloaded swap models only; the Face Swap workspace loads the configured swap
model only when a swap is run.

### Optional face3d Build

InsightFace 1.0.1 does not build the optional ``face3d`` Cython/C++ extension by
default. This keeps the default install lighter and avoids local compiler
requirements. Users who need the legacy mask renderer / face3d path can opt in:

```
pip install -e ".[face3d]" --no-build-isolation --config-settings editable_mode=compat
python setup.py build_ext --inplace --with-face3d
```

The same build can also be enabled with:

```
INSIGHTFACE_WITH_FACE3D=1 python setup.py build_ext --inplace
```

More details:

- ``docs/gui.md``
- ``docs/commercial_evaluation.md``
- ``docs/gui_packaging.md``

## Change Log

### [1.0.1] - 2026-05-23

#### Changed

- Remove the PyPI package metadata license classifier field while keeping the README license guidance.
- Move direct `Pillow` and `scikit-learn` requirements to the GUI extra, and `matplotlib` to the optional `face3d` extra.
- Remove unused base dependencies on `easydict` and `prettytable`.

### [1.0] - 2026-05-23

#### Added

- Add **InsightFace Evaluation Studio**, a cross-platform PySide6 desktop GUI for local face recognition, album grouping, enterprise evaluation/report export, and face swap trials.
- Add GUI launch commands: ``insightface-gui``, ``insightface-eval-studio``, ``insightface-desktop``, and ``python -m insightface.gui``.

#### Changed

- Default ``FaceAnalysis.prepare()`` detection size is now Auto, running SCRFD at both 128x128 and 640x640 before unified NMS.
- Route detection models loaded by ``model_zoo.get_model()`` through ``SCRFD`` by default.
- The optional ``face3d`` Cython/C++ extension is no longer built by default; use ``--with-face3d`` or ``INSIGHTFACE_WITH_FACE3D=1`` to opt in.

### [0.7.1] - 2022-12-14
  
#### Changed
  
- Change model downloading provider to cloudfront.

### [0.7] - 2022-11-28
  
#### Added

- Add face swapping model and example.
 
#### Changed
  
- Set default ORT provider to CUDA and CPU.
 
### [0.6] - 2022-01-29
  
#### Added

- Add pose estimation in face-analysis app.
 
#### Changed
  
- Change model automated downloading url, to ucloud.
 

## Quick Example

```
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)  # Auto detection size: 128x128 + 640x640
img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
```

This quick example will detect faces from the ``t1.jpg`` image and draw detection results on it.



## Model Zoo

In the latest version of insightface library, we provide following model packs:

Name in **bold** is the default model pack. **Auto** means we can download the model pack through the python library directly.

Once you manually downloaded the zip model pack, unzip it under `~/.insightface/models/` first before you call the program.

| Name          | Detection Model | Recognition Model    | Alignment    | Attributes | Model-Size | Link                                                         | Auto |
| ------------- | --------------- | -------------------- | ------------ | ---------- | ---------- | ------------------------------------------------------------ | ------------- |
| antelopev2    | SCRFD-10GF      | ResNet100@Glint360K  | 2d106 & 3d68 | Gender&Age | 407MB      | [link](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing) | N             |
| **buffalo_l** | SCRFD-10GF      | ResNet50@WebFace600K | 2d106 & 3d68 | Gender&Age | 326MB      | [link](https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing) | Y             |
| buffalo_m     | SCRFD-2.5GF     | ResNet50@WebFace600K | 2d106 & 3d68 | Gender&Age | 313MB      | [link](https://drive.google.com/file/d/1net68yNxF33NNV6WP7k56FS6V53tq-64/view?usp=sharing) | N             |
| buffalo_s     | SCRFD-500MF     | MBF@WebFace600K      | 2d106 & 3d68 | Gender&Age | 159MB      | [link](https://drive.google.com/file/d/1pKIusApEfoHKDjeBTXYB3yOQ0EtTonNE/view?usp=sharing) | N             |
| buffalo_sc    | SCRFD-500MF     | MBF@WebFace600K      | -            | -          | 16MB       | [link](https://drive.google.com/file/d/19I-MZdctYKmVf3nu5Da3HS6KH5LBfdzG/view?usp=sharing) | N             |



Recognition Accuracy:

| Name      | MR-ALL | African | Caucasian | South Asian | East Asian | LFW   | CFP-FP | AgeDB-30 | IJB-C(E4) |
| :-------- | ------ | ------- | --------- | ----------- | ---------- | ----- | ------ | -------- | --------- |
| buffalo_l | 91.25  | 90.29   | 94.70     | 93.16       | 74.96      | 99.83 | 99.33  | 98.23    | 97.25     |
| buffalo_s | 71.87  | 69.45   | 80.45     | 73.39       | 51.03      | 99.70 | 98.00  | 96.58    | 95.02     |

*buffalo_m has the same accuracy with buffalo_l.*

*buffalo_sc has the same accuracy with buffalo_s.*



**Note that these models are available for non-commercial research purposes only.**



For insightface>=0.3.3, models will be downloaded automatically once we init ``app = FaceAnalysis()`` instance.

For insightface==0.3.2, you must first download the model package by command:

```
insightface-cli model.download buffalo_l
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
app.prepare(ctx_id=0) # Auto detection size: 128x128 + 640x640

# Method-2, load model directly
detector = insightface.model_zoo.get_model('your_detection_model.onnx')
detector.prepare(ctx_id=0) # SCRFD defaults to Auto: 128x128 + 640x640

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
