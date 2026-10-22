# InsightFace Python Library 2.1

InsightFace provides face detection, recognition, alignment, and attributes
through Python and ONNX Runtime. Optional features include RGB liveness,
PrivateFrame for video face blur/mosaic, PersonAnalysis for person detection and
face/body reference matching, and the Evaluation Studio desktop GUI.

## License

The library code is released under the **MIT License**, for academic and
commercial use. **The pretrained models provided with this library are for
non-commercial research only**, whether downloaded automatically or manually.

## What's new in 2.0

- **[PersonAnalysis update (2.1)](#personanalysis):** find bodies and faces,
  compare them with registered references, and get matching person IDs. Use
  `get()` to extract features, `match()` to compare them, and optional `update()`
  to add body references after a clear face match. Independent face detection
  still works when body detection misses someone. Choose `cheetah_s` or
  `cheetah_l` on CPU or NVIDIA CUDA, or try the GUI workflow with local
  video, a local camera or RTSP and reference photos you can drag into the page.
- **[Liveness update](#optional-liveness-addon):** optional RGB liveness before
  recognition, configurable recognition gating, and per-face results.
- **[PrivateFrame update](#privateframe):** local video face blur/mosaic,
  reference-photo selection, editable analysis JSON, and desktop, CLI, and
  Python API workflows. See the [full guide](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/privateframe/README.md).
- **Runtime and models:** `raccoon_s` / `raccoon_l`, automatic CoreML/CUDA/CPU
  selection, and reusable CoreML compilation caches.

## Installation

Python 3.10 or newer is required.

| Use case | Command |
|---|---|
| FaceAnalysis, PersonAnalysis and ModelZoo | `python -m pip install -U insightface` |
| PrivateFrame API and CLI | `python -m pip install -U "insightface[privateframe]"` |
| Evaluation Studio GUI, including PrivateFrame | `python -m pip install -U "insightface[gui]"` |

The base package installs `onnxruntime`. The `privateframe` extra adds PyAV and
PyYAML; the `gui` extra also includes the Qt desktop application. The optional
`face3d` extension is not compiled by default, so ordinary installation does
not require a C++ compiler. See the
[source installation and runtime guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md)
for installation details.

### NVIDIA CUDA

After installing InsightFace, replace the default runtime with the GPU
distribution:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Do not keep both runtime distributions installed together. Installing or
upgrading InsightFace may install `onnxruntime` again; repeat this replacement
afterward on NVIDIA systems.

## Quick Example

### Detect faces in an image

Detect faces in the bundled sample image and save an annotated image:

```python
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image

app = FaceAnalysis()
app.prepare()
image = get_image("t1")
faces = app.get(image)
cv2.imwrite("t1_output.jpg", app.draw_on(image, faces))
```

`FaceAnalysis()` defaults to `buffalo_l` and downloads the model package on
first use if needed. Models are stored under `~/.insightface/models/` by
default. `prepare()` uses `ctx_id=0` and Auto detection size, combining
128×128 and 640×640 detection.

### Match people in a video

The `cheetah_s` model package downloads on first use if missing. For offline
installation or an unavailable release archive, see the
[model setup guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/person_analysis.md#2-set-up-the-model-package).
Prepare a clear reference face photo `alice.jpg` and a constant-frame-rate
video `clip.mp4`. This CPU example prints each matched ID
or `None` when no registered person matches:

```python
import math
import cv2
from insightface.app import PersonAnalysis

analysis_fps = 2  # Analyze at most twice per video second to speed up processing.
cap = cv2.VideoCapture("clip.mp4")
try:
    if not cap.isOpened():
        raise RuntimeError("Cannot open clip.mp4")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("The video must report a valid frame rate")
    frame_step = max(1, math.ceil(source_fps / analysis_fps))

    with PersonAnalysis(name="cheetah_s") as app:
        registration = app.register("employee_001", "alice.jpg")
        if registration.accepted == 0:
            raise ValueError(f"No reference was accepted: {registration.rejected}")

        frame_index = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                if frame_index < 0:
                    raise RuntimeError("The video opened but did not return a readable frame")
                break
            frame_index += 1
            if frame_index % frame_step:
                continue
            matches = app.match(app.get(frame))
            app.update(matches)  # Optional: learn body references after clear face matches.
            for result in matches:
                print(f"{frame_index / source_fps:.2f}s", result.person_id, result.matched_by)
finally:
    cap.release()
```

A 30 FPS video analyzes every 15th frame. Lower-rate inputs are not duplicated.
The limit applies to **video time**; processing does not wait for playback.
`analysis_fps` belongs to this example's input loop, not `PersonConfig` or
`get()`. For camera/RTSP input with latest-frame handling, use the
[GUI](https://github.com/deepinsight/insightface/blob/master/python-package/docs/gui.md#choose-how-often-to-analyze).
See the [PersonAnalysis guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/person_analysis.md)
for model setup, result fields, all parameters, and video/camera usage.

### Automatic Provider selection

When no provider is specified, FaceAnalysis and ModelZoo select the first available
provider reported by the installed ONNX Runtime:

```text
CoreMLExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider
```

An accelerated provider uses CPU as its fallback when available. Explicit
`providers=[...]` arguments take precedence. CoreML compilation caches are
reused across runs. See the
[runtime guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md)
for provider overrides, CoreML caching, and telemetry behavior.

PersonAnalysis defaults to CPU and also supports explicit CUDA selection. Its
GUI Auto setting chooses CUDA when available, otherwise CPU; it does not use
CoreML.

## PersonAnalysis

PersonAnalysis detects bodies and faces and matches them to IDs you registered.
Use `register()` to add reference photos, `get(frame)` to extract observations,
`match(observations)` to find matching IDs, and optional `update(matches)` to
learn body references after clear face matches. Independent faces can still be
recognized when body detection misses someone. See the
[video example above](#match-people-in-a-video) to get started.

Choose `cheetah_s` for the smaller model package or `cheetah_l` for the larger
one. The API defaults to `cheetah_l` on CPU; CUDA can be selected explicitly.
References stay in memory while the same instance is open. Results describe the
current frame; the API does not create tracks or a history database.

The [complete PersonAnalysis guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/person_analysis.md)
covers installation, model files, video/camera input, registration, face/body
matching rules, result fields, all API arguments, and configuration defaults.
You can also try the [runnable demos](https://github.com/deepinsight/insightface/tree/master/examples/person_analysis)
or use the Person Analysis workflow in the [desktop GUI](https://github.com/deepinsight/insightface/blob/master/python-package/docs/gui.md),
which supports local videos, local cameras, RTSP, reference-photo drag and drop,
and a shared analysis-frequency setting.

## PrivateFrame

PrivateFrame detects and tracks faces in local videos and applies Gaussian
blur or mosaic. Blur all detected faces, blur only people matched to reference
photos, or keep matched people visible. Processing runs locally and preserves
the source video.

```bash
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/output
```

This writes `video_privateframe.mp4` and an editable `video_privateframe.json`.
The default **Fast** mode targets **15 analysis FPS**, including in the GUI;
**Normal (30)** provides denser sampling. Analysis FPS controls detection
sampling, not output FPS: every source frame is rendered. Briefly visible
faces can be missed, so review the result before sharing it.

See the [full guide](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/privateframe/README.md)
for GUI/Python examples, reference photos, JSON editing, configuration,
automation, and a video demo.

## Evaluation Studio GUI

Install `insightface[gui]`, then launch:

```bash
insightface-gui
```

Evaluation Studio includes Person Analysis, PrivateFrame, face comparison and search, People
Library management, album clustering, enterprise evaluation/reporting, and
face swap trials. Workspace data is stored locally and is not uploaded
automatically. See the
[GUI guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/gui.md)
for model downloads, workflows, settings, and troubleshooting.

## Optional liveness addon

Enable RGB liveness explicitly when constructing `FaceAnalysis`:

```python
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(addons=["liveness"])
app.prepare()
image = cv2.imread("input.jpg")
if image is None:
    raise FileNotFoundError("input.jpg")

for face in app.get(image):
    result = face.liveness
    print(result.status, result.is_live, result.live_score)
```

The addon downloads automatically if missing and is verified before loading.
Its default path is `~/.insightface/addons/liveness.onnx`; enabling it does not
require changing the base model package.

The default `liveness_mode="normal"` keeps detected faces in the results but
skips recognition for faces that fail liveness or have rejected input.
`liveness_mode="observe"` continues recognition regardless of that result.
The default live-score threshold is `0.8`. Omitting `addons=["liveness"]`
disables addon downloading, loading, and inference.

See the [liveness guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/liveness.md)
for options, result fields, input rejection, offline setup, and error handling.

## Model Zoo

| Workflow | Default model | Alternatives |
|---|---|---|
| `FaceAnalysis()` | `buffalo_l` | Raccoon packages, other supported legacy packs, or your own compatible models |
| `PersonAnalysis()` | `cheetah_l` | `cheetah_s` |
| PrivateFrame | `raccoon_s` | `raccoon_l` |
| New GUI configurations | `cheetah_s` | Other supported packages for the selected workflow |

Select a package with `FaceAnalysis(name="raccoon_s")`. Model packages live
under `<root>/models/<name>/`; the default root is `~/.insightface`. PrivateFrame
can download its selected Raccoon package on first use. The PersonAnalysis API
and GUI also download a missing selected Cheetah package. Both prefer local
packages; other downloads are initiated from Models. Existing GUI configurations
retain their saved model selection.

See the [model guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md)
for package contents, download links, benchmarks, custom licensed models, and
direct ONNX model calls. Model licenses apply separately from the library's
MIT license.

## Documentation

| Guide | Contents |
|---|---|
| [Runtime and installation](https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md) | Source installs, CUDA, CoreML, provider selection, telemetry |
| [PrivateFrame](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/privateframe/README.md) | Video demo, GUI, CLI, Python API, configuration |
| [Liveness](https://github.com/deepinsight/insightface/blob/master/python-package/docs/liveness.md) | Options, results, offline models, input handling |
| [Evaluation Studio](https://github.com/deepinsight/insightface/blob/master/python-package/docs/gui.md) | Desktop workflows and model management |
| [PersonAnalysis](https://github.com/deepinsight/insightface/blob/master/python-package/docs/person_analysis.md) | Model setup, registration, matching, results, and configuration |
| [Enterprise evaluation](https://github.com/deepinsight/insightface/blob/master/python-package/docs/commercial_evaluation.md) | Datasets, metrics, and reports |
| [Model Zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md) | Model packages and advanced model usage |

## Change Log

See the [complete change log](https://github.com/deepinsight/insightface/blob/master/python-package/CHANGELOG.md)
for the PersonAnalysis update in 2.1, the September 10, 2026 release,
and earlier versions.
