# InsightFace Evaluation Studio: User Guide

InsightFace Evaluation Studio is a desktop application for person detection and matching,
face recognition, video privacy, photo organization, and model evaluation.
Processing and results stay on your computer; the application does not
upload images, videos, embeddings, or reports automatically.

## Open the application

Launch **InsightFace Evaluation Studio** from your installed desktop package.
If you use a Python distribution that includes the GUI, install its GUI
components and launch it with:

```bash
python -m pip install -U "insightface[gui]"
insightface-gui
```

The Python application requires Python 3.10 or newer. Use the application or
package release supplied for your deployment; available features and model
downloads depend on that release.

On first launch, choose a workspace and device setting. New workspaces open
**Person Analysis** with `cheetah_s` selected. Existing workspaces keep their
saved model and last workflow. Use **Workflows** on the left to switch tasks.

The top bar provides three shared controls:

- **Models**: select a model package, model folder, and processing device.
- **Settings**: change the interface language and theme.
- **License**: review model license information and contact options.

## Choose and install models

Open **Models > Runtime**, choose the package for your task, and save the
selection. The chosen package and device appear at the top of the application.

| Workflow | Model selection |
| --- | --- |
| Person Analysis | `cheetah_s` or `cheetah_l` |
| PrivateFrame | `raccoon_s` or `raccoon_l` |
| Face Recognition, Album Management, Enterprise Evaluation | A compatible face package, such as `buffalo_l`, `buffalo_s`, or a Raccoon package |
| Face Swap | A compatible face package and a downloaded face-swap model selected in Runtime |

`cheetah_s` is the smaller person-analysis package; `cheetah_l` uses larger
models. Start with a short representative video to choose the package that
suits your images and computer.

If a non-Cheetah package is selected, the Person Analysis input and result
area is greyed out. **Choose model** above that area remains available: use it
to select `cheetah_s` or `cheetah_l`. A missing or invalid model is described
in the status message below the model name.

### Model folders

The default model root is `~/.insightface`. A model root is the parent of the
`models` folder, not the package folder itself. With the default root, complete
packages belong at:

```text
~/.insightface/models/cheetah_s/
~/.insightface/models/cheetah_l/
~/.insightface/models/raccoon_s/
~/.insightface/models/buffalo_l/
```

To install a supplied package manually, place its complete contents in the
matching folder, preserving its model files and manifest. Select the same
package name in **Models**. For Person Analysis, use the standard Cheetah
package folders rather than the custom face-model directory option.

### Downloads

In **Models > Downloads**, click **Refresh Download URLs**, select an available
package, then click **Download Selected**. After download, use **Use Selected
Model** to make it the current package.

Person Analysis and PrivateFrame can also attempt a download when you start
processing with a supported package whose folder is absent. Download entries
are resolved from the [InsightFace model-zoo release](https://github.com/deepinsight/insightface/releases/tag/model-zoo); an entry does not mean
that every package has already been published. If the requested asset is
unavailable, install the complete package supplied to you in the local folder
shown on the page. An invalid existing Cheetah package is reported instead of
being overwritten automatically.

Stop the current analysis or video task before changing models or devices.
Model downloads and processing must finish before another conflicting model
operation can start.

### Processing device

For **Person Analysis**, **Auto** chooses CUDA when available, otherwise CPU.
Choose **CPU** to use the processor, or **CUDA** to request a compatible NVIDIA
GPU. Person Analysis does not use CoreML. Other face workflows may use CoreML
when Auto is selected on a supported macOS installation.

Person Analysis defaults to 640 for face detection. Body detection defaults to
320 for `cheetah_s` or 640 for `cheetah_l`. **Advanced parameters** provides
separate **Body detection input size** and **Face detection input size**
settings to override those defaults for the next run. Its shared detection-size
control is disabled in Models. In face-only workflows, **Auto** detection size
combines 128 and 640 inputs; other sizes can be selected in Runtime.

## Person Analysis

The first workflow demonstrates the same `get()` → `match()` → `update()` API
available in Python. Choose `cheetah_s` or `cheetah_l` in Models. Until a compatible
model package is selected, the operation panel is disabled and the model picker
remains available. CPU and CUDA are supported; Auto uses CUDA when available.

### Choose input and reference photos

Select a local video, a local camera index, or an RTSP URL. Camera addresses are
used only for the current run and are not saved in settings or reports. Reference
photos are optional. Drag photos into the people area or click Add photos, then
edit the names. Use one clear face per photo; photos with the same name belong to
one person. Face references have no per-person count limit, so three or more
valid, distinct photos can be registered for the same person. Quality and
duplicate checks still apply.

Enable **Automatically add body references** to allow reliable same-frame face
matches to supply body references for later frames. Only a clear face with an
unambiguous body association can add one. A body-only match does not add another
reference. Face references are added only through the reference photos you supply.
Body references default to four per person, counting manual and automatic samples
together. At that limit, only the oldest automatic body sample can be replaced;
manual samples are protected. References are cleared when a run ends; there is no
anonymous-ID memory.

### Choose how often to analyze

**Analyze at most (times/second)** is a shared control in the input section,
above **Advanced parameters** and **Start analysis**. It stays available for
local videos, local cameras and RTSP streams. The default is **0 (Auto)**;
enter a positive value to choose a sampling limit.

| Input | 0 (Auto) | Positive value |
| --- | --- | --- |
| Local video | Analyze every frame in sequence | Sample frames by their timestamps in the video |
| Local camera or RTSP | Analyze the newest pending frame as quickly as possible | Limit analysis starts by elapsed time, using the newest pending frame |

For a 30 FPS video, a value of 5 analyzes approximately every sixth frame,
or five frames per second of video time. There is no delay to match playback
speed: the run can finish faster or slower than the video's duration, depending
on processing speed. Frames between samples are skipped for analysis.

Sampling uses each frame's video timestamp when available. If timestamps are
missing or stop advancing, it falls back to the source video's frame rate. If
neither provides usable timing, a positive limit stops with a message to use
**Auto**. Auto still processes every frame, although its time may be unavailable.

For a camera or RTSP stream, analysis waits when no fresh frame is available.
If processing is slower than the limit, it takes the newest pending frame when
ready and skips older unprocessed frames without building a queue. Time spent
processing counts toward the interval; the GUI does not add a full interval
after each analysis finishes.

The setting never repeats frames to reach a requested rate. Setting a limit
above the source frame rate does not create additional frames.

### Advanced parameters

Open **Advanced parameters** to adjust all 16 `PersonConfig` fields, including
body and face detection input sizes, matching thresholds, face and body quality requirements, body-reference limits,
initial reference capacity and CPU threads. The defaults come from
`PersonConfig()`; see the complete [field and default table](person_analysis.md#11-configuration-and-defaults).
Face references have no per-person count limit.

**Body detection input size** defaults to `0`, meaning the model package's
default: 320×320 for `cheetah_s` or 640×640 for `cheetah_l`. To override it, enter
a positive multiple of 64, such as `640` for a 640×640 input. The setting applies
when starting the next analysis and leaves the model package unchanged. It does
not change the separately configured face detection size or feature extraction
sizes.

**Face detection input size** also defaults to `0`, displayed as **Default
(640)**. This uses the SDK's 640×640 input rather than a value from the model
manifest. Enter a positive multiple of 32, such as `320`, to override it
for the next run. The same input size is used to detect faces in your reference
photos and video or camera frames. This changes the detector input, not the
minimum face-box sizes for recognition (20 original-image pixels) or reference
registration (32 original-image pixels by default). Face recognition still uses
a 112×112 aligned crop, and the package files stay unchanged. Smaller detection
inputs may miss smaller faces; larger inputs do not guarantee better accuracy.
Compare speed and results on your own inputs before changing either size.

**Restore defaults** fills the dialog with the default values. **Save** saves
the changes for the next run; **Cancel** closes the dialog without applying its
edits. Model packages and CPU/CUDA selection remain in **Models**.

### Start and review

Click Start analysis. Green boxes show current matches; amber boxes show unmatched
observations. Labels include the registered name, face/body matching method and
similarity. Associated faces may have an additional face box. Unicode names are
rendered by Qt. No match is inherited from previous frames, so labels may change
when a face or body becomes difficult to recognize.

Local videos process every frame in sequence with Auto, or sample by video
timestamps with a positive limit, and end at EOF. For local cameras and RTSP
streams, a separate reader continuously captures frames while analysis
runs. Only one captured frame can be pending: a newer frame replaces an older
unprocessed frame. Analysis completes `get()` → `match()` → optional `update()`
for the current frame, then takes the latest available frame when the analysis
rate limit permits, or waits for a new one. Camera timestamps record UTC when
OpenCV successfully reads each frame and stay with that frame throughout
analysis. They are not the camera's own capture timestamps. The interface also
keeps only one pending preview result.

Stop ends the input and releases resources after the current native read or
inference finishes. Camera, network and decoder buffers can still add latency;
buffering and read-timeout support depend on the OpenCV backend. This
single-input demonstration does not automatically reconnect a camera, manage
multiple cameras, or save tracks, appearance durations, event logs or a history
database.

### Saved settings

The GUI saves the shared input sampling limit as the top-level setting
`person_analysis_max_fps`, which defaults to `0`. Advanced algorithm overrides
belong to the separate `person_config` dictionary. For example, this GUI
configuration requests up to 15 analyses per second of video time for a file,
or per elapsed second for a camera/RTSP stream:

```json
{
  "person_analysis_max_fps": 15,
  "person_config": {
    "face_similarity_threshold": 0.45,
    "reid_similarity_threshold": 0.85,
    "max_body_samples": 4,
    "reference_capacity": 256
  }
}
```

The dictionary uses the same field names as the SDK; omitted fields use
`PersonConfig()` defaults. `person_analysis_max_fps` controls GUI input sampling.
It is not a `PersonConfig` field or an argument to the SDK's `get()` method, and
does not change the command-line examples. Saved changes take effect when you
start the next run.
Defaults are initial settings, not calibrated accuracy guarantees. See the
[PersonAnalysis guide](person_analysis.md) and [runnable examples](../../examples/person_analysis/README.md).

## Other workflows

### PrivateFrame: blur or mosaic faces in a video

Select `raccoon_s` or `raccoon_l` in Models, then open **PrivateFrame** and add
a video. Choose a privacy policy and **Gaussian** or **Mosaic**. **Fast**
targets 15 analysis FPS and **Normal** targets 30 analysis FPS; these settings
control sampling density, not output-video FPS or guaranteed processing speed.

Choose an output folder and either **JSON only** or **JSON + redacted video**.
The outputs are `<video>_privateframe.json` and, when requested,
`<video>_privateframe.mp4`.

Photo-based privacy options use a reference-photo folder and the largest face
in each photo. **Blur only** blurs matched people; **Exempt** keeps matched
people visible and blurs everyone else. Review the result before sharing it.
The [PrivateFrame guide](../insightface/app/privateframe/README.md) explains
privacy policies, reference photos, and output options in detail.

### Face Recognition: compare a query with a gallery

Select a face model and open **Face Recognition**. Add one image to **Query**
and an image, several images, or a folder to **Gallery**. Click **Run
Recognition**. One gallery image runs a one-to-one comparison; several images
run a ranked gallery search. Review the similarity, threshold, decision, and
face-detection information in the results.

### Album Management: group local photos

Open **Album Management** and add album directories. **Import / Refresh** scans
the images and groups similar faces. Select a group to see its photos, then
double-click a thumbnail to open the original image. Groups are suggestions
for review, not confirmed registered identities.

The similarity threshold defaults to `0.48`; higher values make grouping
stricter. Selected directories and results are saved locally. **Clear** clears
the directory selection while leaving the results visible. **Rebuild All**
asks for confirmation before replacing the saved grouping results.

### Face Swap: choose a source and a target

In **Models > Runtime**, choose a downloaded face-swap model. Open **Face
Swap**, add a source image, and choose an image or video as the target. Click
**Run Face Swap** and review the result. Video output is saved as MP4 in the
exports folder.

Optional GFPGAN restoration is configured under Models and requires its
separately downloaded model. Face-swap and restoration models may have their
own license conditions.

### Enterprise Evaluation: evaluate a labeled dataset

Choose **1:1 Verification** or **1:N Identification**, select the dataset, and
set the multi-face policy. **Auto Split** can divide identity-folder images
into gallery and probe sets; the page describes the required layout. Without
Auto Split, one-to-many evaluation accepts `gallery/<identity>`,
`probe/<identity>`, and optional `unknown/` folders.

Click **Validate Dataset** and resolve any reported problems before **Run
Evaluation**. Results include recognition accuracy, false-accept operating
points, and corresponding thresholds. Export the report for review;
Markdown, HTML, and PDF reports are supported by the GUI installation.

## Workspace, language, and licenses

The default workspace is `~/.insightface/gui`. It contains the local database,
exports, reports, caches, and application logs. To use another workspace when
launching the Python application:

```bash
insightface-gui --workspace /path/to/workspace
```

Use **Settings** to select a language and theme. The default language follows
the operating system when supported, otherwise English. Person Analysis
provides English and Chinese labels; untranslated text in other languages
falls back to English.

Open **License** to inspect the current model's license information. Code and
model files may have different licenses. **Get commercial license** on the
Person Analysis page opens the InsightFace contact page for model
authorization and integration support. Obtain the permissions required for
your model and intended use before commercial deployment.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Person Analysis is greyed out | Click **Choose model** and select `cheetah_s` or `cheetah_l`. |
| Start is disabled | Select a supported model and input. Read the model status message; repair an invalid package before starting. |
| A model download is unavailable | Use **Refresh Download URLs** or install the complete package supplied for your release in the displayed model folder. |
| A reference photo is rejected | Choose a readable photo with exactly one clear face and fill in its name. |
| A camera cannot open | Check its device number or RTSP address, network access, operating-system permissions, and whether another application is using it. |
| A person remains unmatched | Check the reference name, photo and face visibility. A body match needs a body reference: in the GUI, enable automatic body-reference updates so a clear face match can add one; the Python API also supports manual body registration. Face and body matches each need enough similarity and separation from other registered people. |
| Processing is slow | Try `cheetah_s` or an available CUDA device. Compare processing time and recognition results on a representative input. |

For Python installations, CUDA requires a compatible NVIDIA driver and
`onnxruntime-gpu`. If replacing the CPU runtime, use the same environment as
the GUI and keep only one ONNX Runtime distribution installed:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Installing or upgrading `insightface[gui]` can reinstall the CPU runtime, so
check the selected device afterward. You can explicitly start on CPU with
`insightface-gui --provider cpu`. Use `insightface-gui --safe-mode` to open
without automatic model loading when investigating startup problems.

Application logs are at `~/.insightface/gui/logs/app.log`, or the logs folder
of your custom workspace.
