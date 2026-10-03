# PersonAnalysis User Guide

[Python package](../README.md) · [Runnable examples](../../examples/person_analysis/README.md) · [GUI guide](gui.md)

PersonAnalysis finds bodies and faces in video frames or photos, extracts features, and compares them with reference photos you have registered. You choose each person's name or ID, such as `"Alice"`, `"employee_001"`, or the positive integer `1001`.

A **feature** is a list of numbers produced by a model to compare how similar two images are. You supply the images; you do not need to enter or interpret those numbers yourself.

**This guide applies to InsightFace 2.1.**

Start with the [video example](#4-run-your-first-example), then use the
[result fields](#5-read-the-output-and-access-features),
[reference registration](#7-register-multiple-references-and-body-crops),
[configuration tables](#11-configuration-and-defaults) and
[public API reference](#13-public-api-reference) as needed.

Start with these three operations:

| Call | What it does | Changes the reference library? |
| --- | --- | --- |
| `get(image)` | Finds bodies and faces in this image and extracts features for comparison | No |
| `match(observations)` | Compares this image's features with registered references | No |
| `update(matches)` | Adds a body reference for someone matched by their face, when the required conditions are met | Yes; calling it is optional |

The **reference library** means the face and body features kept in memory by the current `PersonAnalysis` object. The SDK does not manage a database, tracks, automatic anonymous identities, or historical events. The examples or your application handle reading video and drawing boxes.

## 1. Install InsightFace 2.1

Use Python 3.10 or newer. Install or upgrade the package with pip:

```bash
python -m pip install -U insightface
```

The installation includes NumPy, OpenCV, and ONNX Runtime for basic inference. The PersonAnalysis Python API does not require a database service or a separate PyAV installation.

You do not need a repository checkout to use the Python API or desktop GUI.
If you specifically want to work with the source code, see
[Install from source](runtime.md#install-from-source).

Start with the default CPU provider. To use an NVIDIA GPU, replace ONNX Runtime after installing InsightFace:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Then pass `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` when creating the object. You also need a working CUDA environment; see [Installation and runtime](runtime.md). Do not keep both ONNX Runtime distributions installed together.

Installing or upgrading InsightFace, including its GUI extra, may install the
CPU runtime again. Repeat the replacement above afterward on NVIDIA systems.

The PersonAnalysis Python API defaults to CPU and supports explicitly selecting CUDA. These two model packages do not use CoreML. Selecting CUDA does not guarantee that every operation runs on the GPU or that a camera stream can be processed in real time.

## 2. Set up the model package

Start with `cheetah_s`. Installing the Python package does not install the model
files. When you first prepare `PersonAnalysis`, a missing `cheetah_s` or
`cheetah_l` package is downloaded from the official
[model-zoo release](https://github.com/deepinsight/insightface/releases/tag/model-zoo).
This also happens when entering `with PersonAnalysis(...)` or first calling
`register()` or `get()`. A complete local package is reused without downloading.

For offline use, download the complete archive on a connected computer and
copy it to your machine. You can also use **Models > Downloads** in the
[GUI](gui.md#downloads). Extract the package so that `manifest.json` is directly
inside this directory, not inside an extra nested `cheetah_s` folder:

```text
~/.insightface/
└── models/
    └── cheetah_s/
        ├── manifest.json
        ├── MODEL.LICENSE
        ├── pp_det_small_320.onnx
        ├── det_500m.onnx
        ├── w600k_mbf.onnx
        └── reid_0265.onnx
```

Here, `~` means your user's home directory. The archive includes `manifest.json`,
the signed `MODEL.LICENSE`, and all four models. Keep these files together;
one ONNX file is not enough to run PersonAnalysis.

| Package | Body detection | Face detection / recognition | Body feature model |
| --- | --- | --- | --- |
| `cheetah_s` | `pp_det_small_320.onnx`, default 320×320 input | `det_500m.onnx` / `w600k_mbf.onnx` | `reid_0265.onnx` |
| `cheetah_l` | `pp_det_med.onnx`, default 640×640 input | `det_10g.onnx` / `w600k_r50.onnx` | The same ReID model |

The package display names are **Cheetah S** and **Cheetah L**. In Python, keep
using `name="cheetah_s"` or `name="cheetah_l"`.

`cheetah_s` uses the same face detection and recognition ONNX files as
`buffalo_s`; `cheetah_l` uses the same files as `buffalo_l`. These face models
are included in each Cheetah package, so you do not need to install Buffalo
separately.

Both body detection and face detection use the full image. Face detection defaults to a 640×640 model input in both packages. The SDK handles resizing, so you do not need to resize the photo first. Returned boxes and face landmarks use coordinates in the original image. Body features are also extracted from body crops taken from that original image.

`face_min_size` and `face_registration_min_size` set minimum usable face-box
sizes in original-image pixels. They do not change the detector's input size.

The package manifest supplies the body detection size; the SDK defaults face
detection to 640×640. You can override them independently when creating an instance with
`PersonConfig(body_det_size=640, face_det_size=320)`, for example. This uses a
640×640 body input and a 320×320 face input without changing the manifest or the
feature extraction models. See [Body detection input size](#body-detection-input-size)
and [Face detection input size](#face-detection-input-size) for accepted values
and how to apply a different size.

The default is `root="~/.insightface"`. For example, if your models are in `D:/models/insightface/models/cheetah_s/`, pass `root="D:/models/insightface"`. Do not point `root` directly at the `cheetah_s` directory.

The SDK checks the manifest, declared model hashes, and preprocessing settings before
loading models. Downloads are checked in a temporary directory before being
installed. An existing incomplete or invalid package is not replaced
automatically; repair that local package using a complete archive. Custom
package names must be installed locally. See the [Model guide](model_zoo.md)
for package selection and setup.

### Model package rules

PersonAnalysis requires four task entries in a V2 manifest: `person_detection`,
`person_reid`, `detection` and `recognition`. Each requires a model `file`.
The package-level fields are `manifest_version: 2`, `model_id`, `license`, and
`tasks`; `display_name` is optional. Additional tasks are allowed. Unknown
package and task metadata are ignored; the supported fields still undergo validation.
In particular, a preprocessing object must follow the rules below.
The task name selects the processing pipeline; an `adapter` field is not needed.
The supplied packages record their normalization settings explicitly:

| Rule | Meaning | Omission rule |
| --- | --- | --- |
| `sha256` | Expected model-file SHA-256; checked when supplied | Optional; included in the supplied Cheetah packages |
| Body detection `input_size` | Model input `[height, width]` | Defaults to `[640, 640]`; Cheetah S declares `[320, 320]` |
| ReID `input_size` | Body-feature input `[height, width]` | Defaults to `[256, 128]` |
| Face detection size | The SDK uses 640×640 unless `PersonConfig.face_det_size` overrides it | Not taken from the detection entry in the manifest |
| Recognition `input_size` / `embedding_dimension` | Aligned face input and output feature length | Existing V2 defaults: `[112, 112]` and `512` |
| Body/ReID `preprocessing` | `(RGB * scale - mean) / std`; `mean` and `std` can be scalars or three RGB values; `scale` is a positive scalar | Missing members use only neutral values: `mean=0`, `std=1`, `scale=1` |
| Face `preprocessing` | `(RGB - mean) / std`, following the existing V2 scalar mean/std format, or `"embedded"` | Existing V2 defaults: detection `mean=127.5, std=128`; recognition `mean=127.5, std=127.5`. If supplied, a numeric object must contain exactly scalar `mean` and positive scalar `std` |

All four tasks also accept `"preprocessing": "embedded"`. This declares that
the ONNX model already contains its numeric normalization. The SDK then sends
raw RGB values in the 0–255 range without applying scale, mean or std a second
time. Resizing, RGB channel order, tensor layout and face alignment still happen
outside the model. The image tensor follows the ONNX input type: float32 or,
for embedded preprocessing, uint8. Numeric normalization requires float32.
The body detector's separate `scale_factor` input remains float32 in both cases.
Use `"embedded"` only when the model was exported with that preprocessing;
changing the manifest does not add normalization to the model.

The supplied Cheetah packages use these settings. Face detection size is set by
the SDK; the other sizes and all normalization values below are in the manifests:

| Task | Input size `[height, width]` | Normalization written in the manifest |
| --- | --- | --- |
| Body detection | Small: `[320, 320]`; large: `[640, 640]` | `mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]` |
| Body ReID | `[256, 128]` | RGB mean approximately `[123.675, 116.28, 103.53]`, std approximately `[58.395, 57.12, 57.375]` |
| Face detection | `[640, 640]` in the SDK, omitted from the manifest | `mean=127.5`, `std=128` |
| Face recognition | `[112, 112]` | `mean=127.5`, `std=127.5` |

All supplied tasks express mean/std against raw 0–255 RGB pixels. Both body
tasks omit scale because its default value `1` is correct; face preprocessing
uses the existing V2 mean/std format without a scale field. The ReID manifest
retains the exact numeric precision used by the model; its rounded values
above are for explanation. The SDK does not infer these normalization
values from the weights. There is no configurable crop-padding step: body
features use the body crop, and face recognition uses landmark alignment.

The current Cheetah models need external normalization, so their manifests
keep the explicit numeric settings above. Raccoon's face-verification task is
an example of an actual model with embedded normalization; that is a different
task from face recognition.

`cheetah_s` and `raccoon_s` use the same `w600k_mbf.onnx` recognition weights;
`cheetah_l` and `raccoon_l` use the same `w600k_r50.onnx` weights. Their respective
`recognition` manifest entries are identical, including
`preprocessing_version: "insightface-arcface-1"`, `input_size: [112, 112]` and
`embedding_dimension: 512`. The declared output dimension is checked against
the loaded model. The packages have different task combinations, so the whole
manifest is not identical.

The existing V2 fields `display_name` (package level), `preprocessing_version`
(face tasks), and `embedding_dimension` (recognition) are also accepted.
Display names do not affect feature compatibility; face preprocessing versions
do. These fields do not add GUI or `PersonConfig` settings.

For ordinary use, keep the supplied model manifest unchanged and use
`PersonConfig` to override body or face detection size when creating an instance.
Model-package rules do not add extra GUI advanced parameters.

## 3. Prepare a reference photo and video

Place a reference photo and a local video in the directory where you will run
the script:

- `alice.jpg`: a reference photo containing exactly one clear face of the person you want to register.
- `clip.mp4`: a constant-frame-rate video to analyze. It may contain several people.

`"employee_001"` is a label you assign to the person in `alice.jpg`. The SDK does not discover a person's real name from a photo. You can use a name in your language, a business ID, or a positive integer. Do not use `None`, an empty string, `0`, or a negative integer.

You do not need to crop the reference down to a face. The SDK detects and aligns it. If multiple faces are detected in a reference image, registration rejects that image instead of choosing someone. By default, the shorter side of a face used for registration must be at least 32 pixels in the original image. A person can look clear in a photo while their face still contains too few pixels.

You can also skip registration. `get()` will still return detected boxes and features, but without references there are no names to match and no anonymous IDs are created automatically.

## 4. Run your first example

Save this code as `first_person_demo.py`. It uses `cheetah_s` on the default CPU
provider and analyzes at most two frames per second of video time. Sampling
reduces model work while keeping the input loop simple.

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

Run `python first_person_demo.py` from the directory where you saved the script.
Output such as `1.00s employee_001 face` means that an observation in the frame
at one second matched that registered ID by face. `None None` means the
observation did not match anyone. There can be several lines for one frame;
a frame with no observations prints no result lines.

`analysis_fps` is a variable in this example's video-reading loop. It is not a
`PersonConfig` field, a `PersonAnalysis` constructor option or a parameter to
`get()`. Keep it positive when changing the example. A 30 FPS video at `2`
uses every 15th frame; a 25 FPS video uses every 13th frame, approximately every
0.52 video seconds. A 5 FPS video with a setting of `15` uses each available
frame once; it does not create or repeat frames.

This simple calculation assumes a constant frame rate. The timestamp printed
is video time, and the loop does not wait for playback: a run can finish faster
or slower than the video's duration. Every frame is still decoded, but only
sampled frames enter the models. For variable-frame-rate files, use actual
frame timestamps; the [GUI workflow](gui.md#choose-how-often-to-analyze) handles
timestamp-based sampling and a frame-rate fallback.

For a live camera, selecting frames is a different task: incoming frames arrive
over elapsed time, and buffered frames can become old while inference runs.
Changing `"clip.mp4"` to `0` does not turn this example into a live loop that
keeps only the latest frame. Use the [camera GUI workflow](#9-run-the-image-video-or-camera-demo)
for continuous capture, replacement of old pending frames, and an elapsed-time
analysis limit. Two analyses per second is a requested limit, not a guarantee
that a model finishes a frame within 500 ms.

The `with` block prepares the models once, keeps references across sampled
frames, and releases this instance's resources when it exits. You can also call
`prepare()` explicitly; it returns `None`, and repeated calls do not reload the
models. Remove the `app.update(matches)` line to disable automatic body-reference
updates. Each output describes the current frame rather than a continuous track.

`register()` supports Unicode reference-image paths. OpenCV handles the video
path; if a video cannot be opened, check its path and codec support on your
system. See the [runnable examples](../../examples/person_analysis/README.md)
for preview windows and other input choices.

## 5. Read the output and access features

### `get(image)` returns observations from the current image

The input must be a nonempty OpenCV BGR image array with shape `(height, width, 3)` and dtype `uint8`. A frame returned by `capture.read()` has this format. `get()` does not accept a file path; decode the frame or photo first. `register()` accepts either paths or image arrays.

Each `Person` has four public fields:

| Field | Meaning | When can it be missing? |
| --- | --- | --- |
| `body_bbox` | Body box `[x1, y1, x2, y2]` in original-image coordinates | `None` when there is no reliably associated body |
| `det_score` | Body detection score | `None` when there is no body box; a face has its own score |
| `reid_feature` | Normalized FP32 feature vector for the current body | `None` when there is no body or it does not meet feature extraction requirements |
| `face` | The associated face result | `None` when there is no reliably associated face |

When `face` is present, its public fields are:

| Field | Value | Meaning |
| --- | --- | --- |
| `bbox` | NumPy array with shape `(4,)` | `[x1, y1, x2, y2]` face box in original-image coordinates |
| `det_score` | Number | Face detection score, separate from identity similarity |
| `kps` | NumPy array with shape `(5, 2)`, or `None` | Five face landmarks used for alignment |
| `embedding` | Normalized FP32 NumPy vector, or `None` | Current face feature; the supplied Cheetah packages use 512 dimensions |

A detected face can still have a box but no feature: in that case, `embedding`
is `None`. A body feature is also optional; check for `None` before comparing or
saving either vector.

A face without a body also produces a `Person`. Its `body_bbox`, `det_score`, and `reid_feature` are all `None`, while `face` is present. A body without a face has `face=None`. Both cases can be passed to `match()`.

If a face could belong to more than one body, the SDK does not force a pairing. It may return separate body observations and a face-only observation. The length of the list is therefore not necessarily the number of distinct people in the image.

### `match(observations)` returns one result per observation

| Field | Meaning |
| --- | --- |
| `observation` | The `Person` object supplied as input |
| `person_id` | The ID you supplied during registration, or `None` when unmatched |
| `matched_by` | `"face"` for a face match, `"body"` for a body match, or `None` when unmatched |
| `similarity` | The cosine similarity accepted for this match, or `None` when unmatched |

**The output has the same count and order as the input. `matches[i].observation` is the original `observations[i]` object, not a new observation.** `match()` does not modify the input observations or the reference library. Read current features from `result.observation.reid_feature` or `result.observation.face.embedding`, checking for `None` first.

Matching follows this order:

1. If there is a valid face feature, compare it with registered face references.
2. If the highest face score and its margin over other people meet the thresholds, return the face match. A body match is not also required.
3. If the face does not match, or there is no face feature, try the body references.
4. If neither matches, return an unmatched result with `person_id`, `matched_by`, and `similarity` all set to `None`.

When a person has several references, their highest score is used. The margin compares different people; two references for the same person are not competitors. Each observation is evaluated independently, so two observations in the same batch can match the same person ID. There is no rule limiting each ID to one use per frame.

This is a comparison of current features. It does not require two frames to confirm a match or inherit an identity from an earlier frame. Similarity is not a probability of being correct: a threshold of `0.85` does not mean 85% accuracy.

### Current features and reference features are different

The vectors returned by `get()` come from the image you just supplied. They are not the vectors of the matched reference photo or a full copy of the reference library.

For example, inside the sampled video loop, you can inspect each result:

```python
for result in matches:
    person = result.observation
    body_feature = person.reid_feature
    face_feature = None if person.face is None else person.face.embedding
    print("Body box:", person.body_bbox)
    print("Face box:", None if person.face is None else person.face.bbox)
    print("Body feature available:", body_feature is not None)
    print("Face feature available:", face_feature is not None)
```

The API does not expose or export the reference library. If your application
needs to keep features, save the outputs of `get()` or `get_reid()` alongside
your own labels and image information. Those outputs describe the supplied
images, not every reference retained by the SDK.

Do not mix vectors from different models. Equal dimensions alone do not make them compatible. When switching between `cheetah_s` and `cheetah_l`, register the photos again for each face model. Check `reid_model_id` before sharing body features. Using `get()` → `match()` on the same instance does not require you to handle these identifiers manually.

## 6. Optionally add body references

A face reference alone does not tell the SDK what someone's clothing looks like. When a clear face matches Alice and its corresponding body is unambiguous, calling `update(matches)` can save that body feature under Alice's ID. The body reference can then help match a later image without a clear face.

`update()` only allows the following:

| Current result | Can it automatically add a body reference? |
| --- | --- |
| Registered person matched by face, unambiguous face/body pairing, valid body feature | Yes, subject to duplicate checks and the body-reference limit |
| Person matched only by body | No; a body match alone cannot repeatedly expand the reference library |
| Face only, without a reliably associated body | No |
| Unmatched or no valid feature | No |

Automatic updates do not add face references or create new identities for strangers. Face references are always added through an explicit `register()` call.

The returned `UpdateResult` has three integer fields:

| Field | Meaning |
| --- | --- |
| `added` | New body-reference samples added |
| `replaced` | Existing automatic body samples replaced by new ones |
| `skipped` | Matches that did not produce a reference change |

Each input match can add at most one body sample, and the three counts add up to the number of input results. These counts describe sample changes, not new people. Duplicate features are skipped. The body-reference limit defaults to four per person, counting manual and automatic samples together. At that limit, only the oldest automatic body sample can be replaced; automatic samples cannot replace manually registered ones.

**For each image, match all observations first, then update the whole batch.** Updates affect future matches. They do not rerun inference or rewrite the results you already received. `update()` requires matches from the same instance. If you register, remove, or clear references in between, match again before updating. Submitting a result that has already been used for an update skips it.

The first video example enables updates. To make this optional, set
`auto_update = True` before its loop, then replace its matching and update lines
with the following block inside the sampled-frame loop:

```python
matches = app.match(app.get(frame))
if auto_update:
    changes = app.update(matches)
    print("Body references:", changes.added, "added,", changes.replaced, "replaced,",
          changes.skipped, "skipped")
```

To establish a body reference, a sampled frame must contain a clear matching
face and an unambiguous corresponding body. A match in a later frame is not
guaranteed: clothing, pose, occlusion, and image quality all affect similarity.
You can also supply frames from different cameras or times to the same `app`;
the SDK does not require camera IDs or timestamps.

References do not expire after a number of minutes within the current instance. Body references are subject to a per-person sample limit; face references have no per-person count limit. This API does not find earlier unknown back views in historical records or calculate first appearance and duration. Your application needs to retain and process that information if required.

## 7. Register multiple references and body crops

You can register multiple face photos under the same person ID to cover different angles. There is no per-person count limit: three or more valid, distinct face samples are accepted in one `register()` call or across repeated calls. Face references are added only by manual registration, and all samples still undergo quality and duplicate checks. The example below uses two photos.

Body references are optional. Supply an already cropped image of one body whose identity you know. The SDK does not run body detection again to choose a person inside `body_images`.

```python
from insightface.app import PersonAnalysis

with PersonAnalysis(name="cheetah_s") as app:
    registration = app.register(
        "Alice",
        ["alice_front.jpg", "alice_side.jpg"],
        body_images=["alice_body_crop.jpg"],
    )
    print("Registered ID:", registration.person_id)
    print("Accepted samples:", registration.accepted)
    for item in registration.rejected:
        print("Rejected:", item["kind"], "image", item["index"] + 1, "reason:", item["reason"])
    if registration.accepted == 0:
        raise ValueError("No reference samples were accepted. Check the reasons above.")
```

`RegistrationResult` has these public fields:

| Field | Value | Meaning |
| --- | --- | --- |
| `person_id` | String or positive integer | The label you supplied |
| `accepted` | Integer | Samples accepted by this call, including faces and manual body references; may include replacement of an automatic body sample, so it is not a count of people or net library growth |
| `rejected` | List of dictionaries | Each rejection has `kind` (`"face"` or `"body"`), zero-based `index` within that kind's input list, and a `reason` string |

A sample may be rejected because it cannot be read, has no face or multiple faces, fails quality checks, or is a duplicate. The default duplicate similarity threshold is `0.98`. The `manual_limit` rejection applies to a body sample when manual body references already fill that person's body-reference limit; face samples are not rejected for reaching a count limit.

If a manually registered body is a near-duplicate of an automatic body sample
for that person, registration can replace the automatic sample and protect it
as a manual reference. It does not add another row. A duplicate of an existing
manual sample is rejected.

Calling `register()` again with the same person ID tries to add references; it does not clear their existing references. To register only a body, supply an empty face list, for example `app.register("Alice", [], body_images="alice_body_crop.jpg")`. Here you explicitly supply the identity label, which differs from the SDK learning automatically from a body match.

To register someone again from scratch, first call `remove_person(person_id)`. It returns the total number of face and body samples removed. `clear_references()` clears everyone's references and returns `None`.

## 8. Draw the results on an image

Each decoded video frame is an image. Define this helper before the video loop
in [the first example](#4-run-your-first-example). Green boxes mean a match was
accepted; orange boxes mean unmatched. When there is no body, it draws the
independent face box. The example uses an ASCII ID because OpenCV's `putText()`
has limited support for non-ASCII text. Use the desktop GUI for labels in other
languages.

```python
import cv2


def draw_matches(frame, matches):
    output = frame.copy()
    for result in matches:
        person = result.observation
        if person.body_bbox is not None:
            box = person.body_bbox
        elif person.face is not None:
            box = person.face.bbox
        else:
            continue

        x1, y1, x2, y2 = [int(round(value)) for value in box]
        matched = result.person_id is not None
        color = (60, 190, 70) if matched else (0, 180, 255)
        label = str(result.person_id) if matched else "unmatched"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output
```

After computing `matches` for a sampled frame, add these lines inside the loop
to save its annotated image. They write one JPEG per analyzed frame:

```python
output = draw_matches(frame, matches)
if not cv2.imwrite(f"frame_{frame_index:06d}.jpg", output):
    raise OSError("Cannot save the annotated frame. Check the output directory.")
```

The colors show the API's matching decision, not correctness verified against human annotations.

## 9. Run the image, video, or camera demo

Use the existing [demo.py](../../examples/person_analysis/demo.py) if you do not want to write the input loop yourself. These optional scripts are in the source repository and are not installed by pip. Download or clone the repository, then run these commands from its root; the library itself can remain installed with the pip command above:

```bash
# Local image: supports Unicode paths, prints results, and displays boxes.
python examples/person_analysis/demo.py --image scene.jpg --reference "Alice=alice.jpg" --display

# Local video: prints current-frame results and displays boxes.
python examples/person_analysis/demo.py --video clip.mp4 --reference "Alice=alice.jpg" --display

# Local camera: 0 is usually the first camera; adjust for your device.
python examples/person_analysis/demo.py --camera 0 --reference "Alice=alice.jpg" --display

# Remote camera: replace this with your own RTSP address.
python examples/person_analysis/demo.py --rtsp "rtsp://camera/stream" --reference "Alice=alice.jpg" --display
```

Add `--gpu` to use CUDA, `--no-update` to disable automatic body references, or `--max-frames 20` to limit processing. Omit `--display` to print results without opening a window, which is useful on a headless machine. You can also run detection without supplying `--reference`.

Press Q or Esc in the video preview, or Ctrl+C in the terminal, to stop. The demo closes its video input and SDK instance. It reads sequentially with OpenCV and calls `get()` → `match()` → optional `update()` for each image. It does not manage multiple cameras, camera reconnection, or real-time queues.

When processing a local video slowly, the demo advances as processing completes. Cameras and RTSP streams may also be affected by device and OpenCV backend buffers, so this demo alone does not guarantee low latency.

If you prefer a desktop interface, install the optional GUI and start it:

```bash
python -m pip install -U "insightface[gui]"
insightface-gui
```

On the first page, Person Analysis, select `cheetah_s` or `cheetah_l`, add reference photos, and choose a local video, local camera, or RTSP stream. Photos support drag and drop, and the page has an option for automatic body references. Labels in other languages can be displayed directly.

The GUI's shared **Analyze at most (times/second)** control appears in the input section above **Advanced parameters** and **Start analysis** for every source type. Its default, **0 (Auto)**, processes every local-video frame in sequence. For a local camera or RTSP stream, Auto analyzes the newest pending frame as quickly as input and processing allow.

For a local video, a positive value samples by timestamps in the video. A 30 FPS video with a value of 2 is analyzed approximately every 15th frame, or twice per second of video time. The GUI does not wait to match playback speed; processing can finish faster or slower than the video's duration. Frames between samples are skipped for analysis.

For a camera or RTSP stream, a positive value limits analysis starts by elapsed time. Capture runs continuously and retains only the newest pending frame. Each `get()` → `match()` → optional `update()` cycle finishes before analysis takes another frame, and processing time counts toward the interval. Slow analysis skips old pending frames without an extra full-interval wait or a queue of old frames. Analysis waits if no fresh frame is available. No input type repeats frames to reach the limit: a 5 FPS source with a limit of 15 still supplies only five distinct frames per second of source time.

The UTC time recorded when OpenCV successfully reads a camera frame stays with that frame. This is a read time, not the camera's own capture timestamp.

Open **Advanced parameters** to edit all 16 algorithm settings listed in [Configuration and defaults](#11-configuration-and-defaults). **Restore defaults** fills in the values from `PersonConfig()`, **Save** saves changes for the next run, and **Cancel** discards the dialog's edits. Choose the model package and CPU/CUDA device in **Models**.

Camera, network and decoder buffers can still add latency. The GUI does not reconnect cameras automatically, and Stop waits for the current native read or inference to finish. The command-line demo above continues to read and process sequentially. See the [GUI guide](gui.md) for desktop usage.

## 10. Extract a feature from a body crop

If you already have an image cropped to one body, call `get_reid()` directly without running body detection:

```python
import cv2
from insightface.app import PersonAnalysis

crop = cv2.imread("person_crop.jpg")
if crop is None:
    raise FileNotFoundError("Cannot read person_crop.jpg. Supply an image cropped to one body.")

with PersonAnalysis(name="cheetah_s") as app:
    feature = app.get_reid(crop)
    print("Feature length:", feature.shape[0])
    print("Data type:", feature.dtype)
```

The current body feature is a normalized, 256-dimensional FP32 vector. The SDK handles the model's size and color preprocessing. Do not pass a whole scene containing several people as one body crop. Invalid or undersized images and invalid feature vectors produce errors.

To compare two body crops, you can also use the existing example:

```bash
python examples/person_analysis/reid.py body_a.jpg body_b.jpg
```

## 11. Configuration and defaults

### Constructor options

`PersonAnalysis(name="cheetah_l", root="~/.insightface", config=None, providers=None)`
accepts four options. Constructing an object does not load the models; entering
its `with` block or calling `prepare()` does.

| Parameter | Accepted value | Default | What to choose |
| --- | --- | --- | --- |
| `name` | Nonempty model-package name as a string | `"cheetah_l"` | Use `"cheetah_s"` for the smaller supplied package or `"cheetah_l"` for the larger one |
| `root` | Nonempty path as a string or `pathlib.Path` | `"~/.insightface"` | The directory containing `models/<name>/`, not the package directory itself |
| `config` | A `PersonConfig` instance or `None` | `None`, which uses `PersonConfig()` | Change algorithm settings listed below; a plain dictionary is not accepted by this constructor |
| `providers` | Nonempty list or tuple of ONNX Runtime providers, or `None` | `None`, which uses `["CPUExecutionProvider"]` | Explicit CPU: `["CPUExecutionProvider"]`; CUDA: `["CUDAExecutionProvider", "CPUExecutionProvider"]` |

Provider entries can also be `(provider_name, options_dict)` tuples when you
need ONNX Runtime provider options. PersonAnalysis supports CPU and CUDA. The
requested runtime must be installed and available; selecting CUDA is not a
request to silently substitute CPU if CUDA cannot initialize. Some model
operations may still execute on the CPU with a CUDA provider chain.

For a first run, choose a model package and keep the algorithm defaults. To
change them, pass a `PersonConfig` to a new instance:

```python
from insightface.app import PersonAnalysis
from insightface.app.person import PersonConfig

config = PersonConfig(
    face_similarity_threshold=0.45,
    reid_similarity_threshold=0.85,
    max_body_samples=4,
    reference_capacity=256,
)
with PersonAnalysis(name="cheetah_s", config=config) as app:
    print("Models are ready. You can register references and process images.")
```

### All 16 algorithm settings

`PersonConfig` accepts these named fields. A **number** below means a finite
Python `int` or `float`; a range of `0–1` includes both endpoints. A positive
integer must be a Python `int` of at least `1`. Booleans, strings, nonfinite
values and unknown fields are rejected. `PersonConfig` is immutable; create a
new configuration and instance when changing settings.

| Option | Type and valid range | Default | Meaning |
| --- | --- | --- | --- |
| `body_det_size` | Integer, `0` or a positive multiple of `64` | `0` | Side length in pixels of the square body detection input; `0` uses the selected package's default: 320 for `cheetah_s`, 640 for `cheetah_l` |
| `face_det_size` | Integer, `0` or a positive multiple of `32` | `0` | Side length in pixels of the square face detection input; `0` uses the SDK default of 640; applies to registration and scene frames |
| `face_similarity_threshold` | Number, `0–1` | `0.45` | Minimum face similarity to a registered reference; raising it requires a closer face match |
| `reid_similarity_threshold` | Number, `0–1` | `0.85` | Minimum body similarity to an existing body reference; raising it requires a closer body match |
| `face_margin` | Number, `0–1` | `0.05` | Minimum gap between the best face candidate and another person; rejects ambiguous face matches |
| `reid_margin` | Number, `0–1` | `0.10` | Minimum gap between the best body candidate and another person; rejects ambiguous body matches |
| `face_min_size` | Positive integer, pixels | `20` | Minimum shorter side of a face box in a scene or video frame for face features |
| `face_registration_min_size` | Positive integer, pixels | `32` | Minimum shorter side of a face box used to register a reference |
| `face_min_score` | Number, `0–1` | `0.6` | Minimum face detection score before extracting a face feature |
| `body_min_size` | Positive integer, pixels | `16` | Minimum shorter side of a body box or manual body crop for body features |
| `body_min_score` | Number, `0–1` | `0.5` | Minimum body detection score before extracting a body feature |
| `face_body_margin` | Number, `0–1` | `0.12` | Required clarity of the face/body pairing; helps avoid attaching a face to an ambiguous body |
| `max_body_samples` | Positive integer | `4` | Maximum body references per person, counting manual and automatic samples together |
| `reference_capacity` | Positive integer | `256` | Initial reserved rows in each reference matrix; grows as needed and is not a limit on people or face references |
| `duplicate_similarity_threshold` | Number, `0–1` | `0.98` | Similarity at which samples for the same person are treated as duplicates |
| `cpu_threads` | Positive integer | `4` | CPU thread setting for each inference session; actual speed depends on the model and hardware |

Minimum face and body sizes refer to boxes in the original image, not the size after internal resizing. `body_det_size` and `face_det_size` instead control the corresponding detector's input image size. Faces also need five valid landmarks for alignment. Detection scores and identity similarity scores are different values; lowering one does not automatically change the other. `face_min_score` and `body_min_score` control feature extraction for detected objects. They do not replace the model package's own detection filtering settings.

These are starting values, not optimal settings or calibrated error rates. Lowering a similarity threshold may reduce unmatched results while increasing incorrect matches. Check scenes with similar clothing, people crossing, distant subjects, and occlusion.

### Body detection input size

Leave `body_det_size=0` to use the selected model package's default. To override
that default, pass the size when creating `PersonAnalysis`:

```python
from insightface.app import PersonAnalysis
from insightface.app.person import PersonConfig

with PersonAnalysis(
    name="cheetah_s",
    config=PersonConfig(body_det_size=640),
) as app:
    print("Body detection uses a 640 x 640 input for this instance.")
```

The value is one integer for both height and width, not a `(width, height)` pair.
Use `0` for the package default or a positive multiple of 64, such as `320` or
`640`. Negative values, floats, booleans and other integers are rejected.
The requested size must also be supported by the model in the package.

The override is applied before model preparation and warmup. It affects this
instance only and does not rewrite the package's `manifest.json`. To change it,
close the instance and create another with a new configuration; there is no
mid-run size setter. In the GUI, set **Body detection input size** in **Advanced
parameters**, save it, and start a new analysis.

This option does not change the separately configured face detection size, face recognition
(112×112), or body feature extraction (256 pixels high by 128 pixels wide).
A larger body input can change both detections and processing time; it does not
guarantee better accuracy. Check the speed and results on your own video before
choosing an override.

### Face detection input size

Leave `face_det_size=0` to use the SDK default of 640×640. This value comes from
the code, not from the package manifest. Set a positive multiple of 32 when
creating the instance to try another square input, such as 320×320:

```python
from insightface.app import PersonAnalysis
from insightface.app.person import PersonConfig

with PersonAnalysis(
    name="cheetah_s",
    config=PersonConfig(face_det_size=320),
) as app:
    print("Face detection uses a 320 x 320 input for this instance.")
```

The value is one integer for both height and width. `160`, `320` and `640` are
examples of valid configuration values; the model must support the requested
input. Negative values, floats, booleans and integers that are not multiples of
32 are rejected. This setting is applied before model preparation and warmup,
and the same size is used when detecting faces in reference photos during
`register()` and in scene or video frames during `get()`.

It does not change the face recognition model, its 112×112 aligned input, or the
body detector and ReID model. It also does not change the minimum original-image
face size: `face_min_size=20` for scene frames and
`face_registration_min_size=32` for reference photos by default. For example,
setting `face_det_size=320` changes the image sent to the detector; it does not
mean that a face must be 320 pixels wide to be recognized.

The override belongs to this instance and leaves `manifest.json` unchanged.
Create a new instance to change it. In the GUI, set **Face detection input size**
in **Advanced parameters**, save it, and start a new analysis. The GUI displays
zero as **Default (640)**. Smaller inputs
can reduce computation but may miss smaller faces or change detected landmarks;
larger inputs do not guarantee better accuracy. The default remains 640×640 in
both packages. Compare results and speed before choosing another size.

### SDK JSON configuration

`PersonAnalysis.from_config(source, **overrides)` accepts a JSON-file path
(`str` or `Path`) or a dictionary with these constructor keys: `name`, `root`,
`providers`, and `config`. Its `config` dictionary is converted to
`PersonConfig`. Omitted options use the constructor defaults; unrecognized keys
are rejected. The supplied [person_config.json](../../examples/person_analysis/person_config.json)
uses this format:

```json
{
  "name": "cheetah_s",
  "root": "~/.insightface",
  "providers": ["CPUExecutionProvider"],
  "config": {
    "body_det_size": 0,
    "face_det_size": 0,
    "face_similarity_threshold": 0.45,
    "reid_similarity_threshold": 0.85,
    "max_body_samples": 4,
    "reference_capacity": 256
  }
}
```

Load it instead of specifying the constructor settings in Python:

```python
from insightface.app import PersonAnalysis

with PersonAnalysis.from_config("examples/person_analysis/person_config.json") as app:
    print("Models are ready using the JSON configuration.")
```

Run this example from the repository root. A relative `root` in a JSON file is resolved relative to that file's directory. `~/.insightface` still refers to the current user's default model directory.

Keyword overrides take precedence, for example
`PersonAnalysis.from_config(path, name="cheetah_l")`. Overriding `config`
replaces that option as a whole; its omitted fields return to `PersonConfig`
defaults. An explicitly overridden relative `root` uses the current working
directory, just as it does in the direct constructor.

### GUI settings

The GUI saves algorithm overrides under `person_config` in its own settings
JSON; omitted fields use the defaults in the table above. Its shared input
sampling limit is the separate top-level key `person_analysis_max_fps`, with a
default of `0` for Auto. For example:

```json
{
  "person_analysis_max_fps": 2,
  "person_config": {
    "face_similarity_threshold": 0.45,
    "reid_similarity_threshold": 0.85
  }
}
```

This is GUI settings JSON, not the configuration format accepted by
`PersonAnalysis.from_config()`. The limit controls only GUI input sampling,
using video timestamps for files and elapsed time for live cameras. It is not a
`PersonConfig` field or an argument to the SDK's `get()` method. The command-line
examples continue to process sequentially. GUI settings changes apply to the
next run.

## 12. Reference lifetime and storage

Faces and bodies each have an FP32 matrix in memory. Matching computes only over references that have been written, not unused reserved rows. Once the initial capacity fills up, it grows by doubling, which can introduce a memory allocation and copying cost.

Face references have no per-person count limit and are added only through `register()`. All valid, distinct face samples are retained, including three or more registered in one call or across repeated calls. Quality checks still apply, and samples at or above the duplicate similarity threshold (`0.98` by default) are rejected as duplicates.

Body references default to at most four per person in total, including manual and automatic samples. Automatic updates skip near-duplicates; manual registration can promote an automatic sample as described in [Reference registration](#7-register-multiple-references-and-body-crops). When adding a distinct sample at the body-reference limit, only that person's oldest automatic body sample can be replaced. Manually registered samples are protected. If manual body samples already fill the limit, a new distinct manual body sample is rejected with `manual_limit`, and automatic body updates are skipped. You can increase `max_body_samples` when creating the instance or remove and register that person again with different samples.

References in the current instance are released when the program exits, the `with` block ends, or you call `close()`. There is no automatic disk storage, restart recovery, image archive, or history query. Reuse the same `app` across images or video frames; do not create and close a new instance for every frame.

It is safe to call `close()` more than once. A closed instance cannot run inference or matching; create a new one. Observations you already received can still be read or drawn, but they cannot be submitted to the closed instance to update references.

## 13. Public API reference

Import the main class with `from insightface.app import PersonAnalysis` and
settings with `from insightface.app.person import PersonConfig`. The result
classes `Face`, `Person`, `MatchResult`, `RegistrationResult` and `UpdateResult`
are also available from `insightface.app.person`.

The following table lists every public method on `PersonAnalysis`. Input image
arrays use the nonempty BGR `uint8` format described in
[Read the output](#5-read-the-output-and-access-features).

| Call | Parameters | Return value |
| --- | --- | --- |
| `PersonAnalysis.from_config(source, **overrides)` | JSON-file path or dictionary; optional overrides for `name`, `root`, `providers`, `config` | A new, unprepared `PersonAnalysis` instance; see [SDK JSON configuration](#sdk-json-configuration) |
| `prepare()` | None | `None`; loads models once and initializes empty in-memory references |
| `get(image)` | One decoded scene image or video frame as a BGR `uint8` NumPy array | `list[Person]` with boxes and current features; no identity matching or reference updates |
| `get_reid(crop)` | One already cropped body as a BGR `uint8` NumPy array | A normalized FP32 NumPy feature vector; 256 dimensions for the supplied Cheetah packages |
| `register(person_id, images, *, body_images=None)` | Identity label, face-reference images, and optional body-reference crops; details below | `RegistrationResult` with `person_id`, `accepted`, `rejected` |
| `match(observations)` | An iterable of `Person` observations returned by `get()` with compatible feature models | `list[MatchResult]`, one per input in the same order |
| `update(matches)` | An iterable of original `MatchResult` objects from this instance | `UpdateResult` with integer counts `added`, `replaced`, `skipped`; [update rules](#6-optionally-add-body-references) apply |
| `remove_person(person_id)` | A valid string or positive integer identity label | Integer count of removed face and body reference samples; `0` if none existed |
| `clear_references()` | None | `None`; removes every face and body reference |
| `close()` | None | `None`; releases the instance's models and references; repeated calls are safe |

`register()` parameters are:

| Parameter | Accepted value | Requirement |
| --- | --- | --- |
| `person_id` | A nonblank Python string or positive Python integer | Required; booleans are rejected; the same ID combines references for one person |
| `images` | A reference-image path (`str` or `Path`), BGR `uint8` array, or iterable of those values | Required argument; each face photo must contain exactly one detected, usable face; use `None` or `[]` for body-only registration |
| `body_images` | A body-crop path, BGR `uint8` array, or iterable of those values | Optional, default `None`; keyword-only; each image must already be cropped to the person you label |

Supply at least one face image or body crop. Check `accepted` and `rejected`:
one rejected image does not discard other accepted samples. See
[Register multiple references](#7-register-multiple-references-and-body-crops)
for an example and rejection details.

All analysis and reference methods load the models on first use if necessary.
`with PersonAnalysis(...) as app:` calls `prepare()` on entry and `close()` on
exit, including when the block raises an exception. It keeps the same instance
throughout the video loop; creating a new instance for each frame would reload
models and lose references.

After preparation, `app.face_model_id` and `app.reid_model_id` identify the
feature models. Equal vector lengths do not establish compatibility.
`match()` checks feature-model identifiers and rejects incompatible features.
`update()` additionally requires the original results from the same instance;
rematch after successful registration, reference removal or clearing.

Every public result type provides `to_dict()`. It returns the public fields,
converts NumPy arrays to lists and nested result objects to dictionaries, and
leaves out private bookkeeping. For example, inside the video loop:

```python
for result in matches:
    row = result.to_dict()
    print(row["person_id"], row["observation"]["body_bbox"])
```

These dictionaries are convenient for your own output or storage. They are
not a saved reference library and cannot replace `MatchResult` objects when
calling `update()`.

## Licensing

The SDK code follows the project's MIT License. Supplied pretrained models have their own licensing terms and are intended for non-commercial research by default. The code license does not grant commercial use of the models. For commercial deployment, check the terms supplied with your models or contact InsightFace through the [contact page](https://www.insightface.ai/contact).
