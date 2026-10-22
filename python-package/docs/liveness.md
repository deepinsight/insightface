# Optional liveness addon

[Back to the package README](../README.md#optional-liveness-addon)

`FaceAnalysis` can run RGB liveness detection before recognition. Enable it
explicitly; installing an addon file alone does not change existing behavior.
The model addon is distributed separately from the base models; see the
[model addons release repository](https://github.com/deepinsight/insightface-model-addons)
for provenance and applicable notices.

## Usage

```python
import cv2
from insightface.app import FaceAnalysis

image_bgr = cv2.imread("input.jpg")
if image_bgr is None:
    raise FileNotFoundError("Could not read input.jpg")

app = FaceAnalysis(
    name="buffalo_l",
    addons=["liveness"],
    liveness_mode="normal",
    liveness_threshold=0.8,
)
app.prepare(ctx_id=0)
faces = app.get(image_bgr)

for face in faces:
    result = face.liveness
    if result is None:
        print("Liveness was not run")
    elif result.status == "input_rejected":
        print(result.reason)
    elif result.is_live:
        print("Live:", result.live_score)
    else:
        print("Fake:", result.live_score)

    # None when recognition was not selected or was blocked by normal mode.
    embedding = face.embedding
```

The three keyword-only options are:

| Option | Behavior |
| --- | --- |
| `addons=["liveness"]` | Select the liveness addon independently of `allowed_modules`. Omit it or use `addons=[]` to disable liveness: no addon download, loading or inference, and existing recognition behavior is unchanged. |
| `liveness_mode` | `normal` (default): recognize only faces whose liveness result is `True`. `observe`: run liveness and continue recognition regardless of its classification or input rejection. This option only takes effect when the liveness addon is selected. |
| `liveness_threshold` | Live-score threshold in `[0, 1]`, default `0.8`; equality passes. |

## Download and offline installation

The addon is downloaded from the
[InsightFace model addons Release](https://github.com/deepinsight/insightface-model-addons/releases/download/addons/liveness.onnx)
to **`<root>/addons/liveness.onnx`**, default `~/.insightface/addons/liveness.onnx`.
All addon files use this flat directory. Downloads are verified against the
SHA256 in the packaged addon catalog before installation; cached files are also
verified before loading.

For offline use, place the published file at this path before constructing
`FaceAnalysis`. An existing file with an unexpected digest raises an error and
is not overwritten.

## Results and recognition behavior

`get()` still returns a list of `Face` objects. Each evaluated face has a
`liveness` result with three core fields, accessible as attributes or dictionary
keys:

| Result | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Live | `"ok"` | `True` | Model probability |
| Fake | `"ok"` | `False` | Model probability |
| Input rejected | `"input_rejected"` | `None` | `None` |

When the liveness addon is not selected, the `liveness` key is absent and
`face.liveness` returns `None`, following the existing `Face` attribute convention.
No detected faces still returns `[]`. Fake and rejected faces remain in the list,
with their bounding boxes and landmarks; normal mode skips only the recognition
task. Other selected tasks retain their existing behavior.

### Input rejection

Only insufficient source-image area around the aligned face produces
`input_rejected`. It adds a human-readable `reason`; live and fake results omit
this field. `FaceAnalysis` always returns this English text:

> Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image.

Use `status` and `is_live` for program logic, not the wording of `reason`. For
older results without it, a client can use
`result.get("reason") or "Input rejected by liveness detection."`.

### Errors

Invalid landmarks raise `ValueError`; an alignment failure raises `RuntimeError`.
These errors, model-loading errors, inference failures and invalid model outputs
raise exceptions in both `normal` and `observe`, ending the call; they are never
reported as fake or silently ignored. Recognition and other models still require
their own valid inputs in observe mode.

## Preprocessing and threshold selection

The adapter accepts the original BGR image and detector five-point landmarks.
It uses a dedicated fixed 80x80 alignment template, rejects aligned crops with
more than 30% missing source area, and fills accepted crop borders by replication.
The model receives RGB float32 NCHW pixels divided by 255 and directly outputs a
live probability. The model's alignment is separate from ArcFace alignment.

Scores can differ across execution providers; validate the operating threshold
with the provider used in deployment. The initial threshold and crop gate are
integration defaults, not a production accuracy guarantee.
