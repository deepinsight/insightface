# PrivateFrame

PrivateFrame detects and tracks faces in local videos and applies Gaussian blur
or mosaic before you share them. Use it from the InsightFace desktop GUI, the
command line, or Python. The source video stays unchanged; processing produces
a separate video and a reusable analysis JSON.

- Blur every detected face, blur only people matched to reference photos, or
  keep matched people visible while blurring everyone else.
- Track face regions across frames and stabilize their geometry for rendering.
- Analyze once, inspect or edit the face boxes in JSON, and render again without
  rerunning detection or recognition.
- Adjust detection sampling, redaction strength, video encoding, and audio.
- Integrate with scripts using command discovery, dry runs, structured status,
  and progress events.

> **Commercial use:** Using PrivateFrame with InsightFace pretrained models for commercial purposes requires a commercial model license. For licensing, visit [www.insightface.ai](https://www.insightface.ai).

## Video demo

https://github.com/user-attachments/assets/2eaa0d92-7d77-4e08-9be5-cbdf1ba50f50

Music: [Tears in Rain](https://www.scottbuckley.com.au/library/tears-in-rain/) by Scott Buckley · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Excerpted and mixed.

## Installation

Install or upgrade InsightFace with the extra for your workflow:

```bash
# PrivateFrame CLI and Python API
python -m pip install -U "insightface[privateframe]"

# Desktop GUI, including PrivateFrame
python -m pip install -U "insightface[gui]"
```

The `privateframe` extra installs PyAV and PyYAML alongside the base InsightFace
dependencies. The `gui` extra also installs the Qt desktop application. See the
[Python package README](../../../README.md#installation) for ONNX Runtime and
GPU installation instructions.

```bash
insightface-privateframe --help
# Equivalent entry point:
python -m insightface_privateframe_bootstrap --help
```

PrivateFrame uses the manifest-backed `raccoon_s` or `raccoon_l` model package;
the default is `raccoon_s`. An absent package can download on first execution to
`~/.insightface/models/<package>`. With a custom `models.root`, the package must
be under `<root>/models/<package>`; that root is authoritative. An invalid
installed package is rejected. Once dependencies and models are installed,
video processing runs locally without uploading the video or reference photos.

## Desktop GUI

Launch the application with:

```bash
insightface-gui
```

1. Open **PrivateFrame** in the **Workflows** rail. Select `raccoon_s` or
   `raccoon_l` in the application's global model settings; PrivateFrame shares
   the global model root and execution provider.
2. Select a local video and an output directory. The input card shows a preview
   and video metadata.
3. Keep **Fast (default, target 15 analysis FPS)** or choose **Normal (target 30
   analysis FPS)** for denser detection sampling. Select who to blur and
   **Gaussian blur** or **Mosaic**. For either photo mode, select a folder
   containing reference photos.
4. Choose video output or **Analysis only (render video later)**. Every run
   writes an analysis JSON; video output also writes the paired MP4.
5. Start processing. Progress and cancellation are available while the job runs
   in the background. **Processing details** shows the live log and output
   filenames. Additional controls include the redaction margin and encoding
   quality.

**Preserve original audio when supported** copies existing AAC audio. The GUI
omits other audio formats and reports that the result will be silent. To
transcode other audio formats, use the CLI's FFmpeg backend as described below.

## Command-line quick start

| Command | Input | Output |
|---|---|---|
| `process` | Source video, optional configuration and reference photos | Analysis JSON and redacted video |
| `analyze` | Source video, optional configuration and reference photos | Analysis JSON only |
| `render` | Source video and existing analysis JSON | Redacted video, without model inference |

```bash
# Blur every detected face; writes video_privateframe.json and .mp4.
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/output

# Analyze first, so you can inspect or edit the JSON before rendering.
insightface-privateframe analyze \
  --input /data/video.mp4 --output-dir /data/review

# Read /data/review/video_privateframe.json and render the paired MP4.
insightface-privateframe render \
  --input /data/video.mp4 --output-dir /data/review
```

`--output-dir` derives names from the input stem. Use `--result` for an explicit
JSON path and `--redacted` for an explicit video path; these override the derived
paths. `render` always needs the original source video as well as the JSON.
Existing public output files are protected by the CLI; add `--overwrite` only
when replacing those files is intentional.

## Choosing who to blur with reference photos

The default `recognition.mode: all` needs no reference photos and does not load
the face recognizer. The two selective modes compare tracked faces against a
folder of photos:

| Mode | People matched to reference photos | Unmatched or uncertain people with `unknown_action: auto` |
|---|---|---|
| `all` | Blurred; photos are not read | Blurred |
| `blur_only` | Blurred | Kept visible |
| `exempt` | Kept visible | Blurred |

```bash
# Blur only the people shown in reference photos.
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/selected \
  --recognition.mode blur_only \
  --recognition.reference_dir /data/reference_photos

# Keep people shown in the photos visible; blur everyone else.
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/exempt \
  --recognition.mode exempt \
  --recognition.reference_dir /data/reference_photos
```

Put JPG, JPEG, PNG, or WebP files directly in the reference folder. Different
people and multiple photos of one person can share the folder; no person names
or per-person subfolders are required. Hidden files and symlinks are skipped,
and identical copies are deduplicated.

Each photo contributes **only its largest detected face**. Prefer clear photos
of one person. If the largest face is unsuitable for recognition, the photo is
skipped with a reason rather than switching to a smaller face. Import messages
report used and skipped photo counts, not the number of people. Both selective
modes stop before video analysis if no reference photo supplies a usable face.
Usable references with no matches in the video are a valid completed result.
Model loading or inference failures stop processing.

Keep `recognition.unknown_action: auto` for the behavior in the table. Set `blur`
or `keep` explicitly only to change the fallback for unmatched or uncertain
faces. **With `keep`, including the default `blur_only` policy, a target person
who is not recognized can remain visible.** In `all` mode, every detected face
is blurred regardless of this setting.

The result JSON stores matching decisions and the effective rendering policy.
Later rendering does not need the reference photos and does not rerun matching.
An analysis created in `all` mode has no recognition decisions; rerun analysis
with a photo mode when you need selective matching.

## Analysis sampling rate

`scan.max_analysis_fps` defaults to **15**, the **Fast** mode also selected by
default in the GUI. **Normal (30)** provides denser sampling. The setting is a
soft ceiling on regular full-frame detection along the video's timeline,
measured in frames per second **of input video**. It is not output FPS or a
processing-speed target; all source frames are still decoded and rendered at
the source frame rate.

| Source FPS | Fast: default target 15 | Normal: target 30 |
|---|---|---|
| 25 | Every 2 frames: 12.5 scans/s | Every frame: 25 scans/s |
| 30 | Every 2 frames: 15 scans/s | Every frame: 30 scans/s |
| 60 | Every 4 frames: 15 scans/s | Every 2 frames: 30 scans/s |
| 15 or below | Every frame | Every frame |

Sampling uses a uniform integer interval with a 5% rate tolerance, so nearby
settings may produce the same interval. Scene changes, video endpoints, and
new tracks can add scans above this soft ceiling. By default, existing face
regions between regular detections are interpolated; a face visible only
between sampled frames may be missed.

The default Fast (15) reduces detector work, especially for scenes with limited
motion. Choose Normal (30) for denser sampling with fast motion, brief
appearances, or frequent occlusion. Lowering to 10 reduces detector work further
but widens sampling gaps. For higher-FPS input, raise toward the source FPS to
scan every frame. Higher sampling costs more compute and still cannot guarantee
that every face is detected.

```bash
# Choose denser sampling instead of the default Fast mode.
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/normal \
  --scan.max_analysis_fps 30
```

## Configuration and rendering

`analyze` and `process` use the packaged [Base configuration](configs/base.yaml).
A custom YAML passed with `--config` inherits that Base, so it only needs the
fields you want to change:

```yaml
schema_version: 1
scan:
  max_analysis_fps: 30  # Denser sampling than the default Fast mode (15).
render:
  redaction:
    method: mosaic
    box_scale: 1.15
```

```bash
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/custom \
  --config /data/privateframe.yaml
```

CLI dotted options are applied last, after the Base and custom YAML. An explicit
`base_config` can select a different complete parent configuration. For `render`,
use `--render-config` with a rendering-only YAML or `--render.*` dotted options;
these override the defaults saved in the JSON. Detection, tracking, and
recognition changes require a new analysis.

| Setting | Default | Purpose |
|---|---|---|
| `models.name` | `raccoon_s` | Select `raccoon_s` or `raccoon_l`. |
| `models.root` | `~/.insightface` | Set the InsightFace root containing `models/`. |
| `runtime.provider` | `auto` | Select an available ONNX Runtime execution provider. |
| `scan.max_analysis_fps` | `15` | Fast mode by default; use 30 for denser sampling. |
| `recognition.mode` | `all` | Choose the reference-photo policy. |
| `recognition.profile` | `balanced` | Choose `fast`, `balanced`, or `accurate` matching, using up to 1, 3, or 5 eligible frames per track. |
| `recognition.unknown_action` | `auto` | Set how unmatched or uncertain faces are treated. |
| `render.redaction.method` | `gaussian` | Choose Gaussian blur or `mosaic`. |
| `render.redaction.box_scale` | `1.0` | Scale the final face region; values above 1 add margin. |
| `render.video_output.encoder` | `libx264` | Select an encoder supported by the installed backend. |
| `render.video_output.preset` | `medium` | Adjust encoding speed and compression. |
| `render.video_output.rate_control.quality` | `23` | Set CRF quality: 18 retains more detail; 28 produces smaller files. |
| `render.video_output.audio.redacted` | `aac` | Choose `none`, `copy`, or `aac`. |

`auto` checks the installed ONNX Runtime and selects CoreML, then CUDA, then CPU
in that order, using the first available accelerator with CPU fallback where
available. Set, for example, `--runtime.provider CPUExecutionProvider` to select
CPU explicitly. A provider must be available in your installed runtime; see the
[provider installation guide](../../../README.md#automatic-provider-selection).

Redaction and encoding can be changed without repeating analysis:

```bash
insightface-privateframe render \
  --input /data/video.mp4 \
  --result /data/review/video_privateframe.json \
  --redacted /data/review/video_mosaic.mp4 \
  --render.redaction.method mosaic \
  --render.redaction.box_scale 1.15 \
  --render.video_output.rate_control.quality 18
```

Gaussian strength is controlled by `render.redaction.gaussian.*`; mosaic block
size by `render.redaction.mosaic.*`. Edge feathering is disabled by default.
Smaller coverage, weaker redaction, or transparent edges can expose facial
detail. Encoding quality affects compression and file size, not face detection
or recognition. Full fields, defaults, and constraints are listed in the
[configuration reference](docs/configuration.md).

### Audio and video compatibility

The default in-process `pyav` backend writes H.264 MP4 with `yuv420p`, CRF 23,
and the `medium` preset. Encoder availability depends on the installation;
use `--dry-run` to check the requested settings.

For CLI/API use, the default `audio.redacted: aac` preserves existing AAC audio
without re-encoding. With the PyAV backend, non-AAC audio requires an explicit
choice: `none` omits audio, while `copy` remuxes it if the output container
supports the source codec. The default AAC setting rejects non-AAC source
audio. To transcode it to AAC, install FFmpeg and select its backend:

```bash
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/with_audio \
  --render.video_output.backend ffmpeg \
  --render.video_output.audio.redacted aac
```

Audio is preserved or omitted, not anonymized. Rendering uses a constant frame
rate derived from the source metadata and decoded frame indices; it does not
preserve individual variable-frame-rate timestamps. Review timing and audio
synchronization when working with variable-frame-rate material.

## Output files and editable JSON

For `/data/video.mp4` with `--output-dir /data/output`, the public artifacts are:

```text
/data/output/video_privateframe.json
/data/output/video_privateframe.mp4
```

Analysis uses `/data/output/.video_privateframe_work` for temporary work; its
encoded-packet cache is removed after analysis. `--workdir` overrides that
location. When using only `--workdir`, without `--output-dir` or `--result`,
the JSON defaults to `<workdir>/result.privateframe.json`.

The portable result document has `format: "privateframe-result"` and
`schema_version: 1`. Its main fields are:

| Field | Contents |
|---|---|
| `source_video` | Source filename, dimensions, FPS, decoded frame count, and coordinate/timing conventions. |
| `observations` | Final face regions: zero-based `frame_idx`, `track_id`, `box` as `[x1, y1, x2, y2]` in pixels, and `source`. Optional flags mark reduced assurance or unconfirmed identity. |
| `recognition` | Whether matching ran, accepted reference records, and per-track matching decisions. |
| `render_defaults` | Redaction, video/audio settings, and effective `recognition_policy`. |

The JSON contains no image pixels, absolute source path, source-content hash,
or face embeddings. It can include reference filenames and matching decisions,
so consider those contents before sharing it.

To correct coverage, copy the JSON, edit the relevant per-frame `box` values,
and render the edited file with `--result`. Keep its source metadata and valid
frame indices intact; use `x2 > x1` and `y2 > y1`. Removing an observation removes
that region's redaction unless another observation covers it. Editing a box on
one frame does not propagate it to other frames. Inspect adjacent frames too.

Rendering validates the document and checks source dimensions, FPS, and decoded
frame count. It does **not** hash source content: a different video with the same
geometry and timing may pass those checks. Always pair the JSON with its
original source video. You can change the stored rendering policy, but selective
rendering requires recognition decisions from an analysis that used photos.

## Python API

Use the exported functions to analyze and render separately:

```python
from insightface.app.privateframe import (
    analyze_streaming_pipeline,
    default_output_paths,
    render_streaming_artifacts,
)
from insightface.app.privateframe.base_config import DEFAULT_CONFIG_PATH

paths = default_output_paths("/data/video.mp4", "/data/python_output")

analysis = analyze_streaming_pipeline(
    config_path=DEFAULT_CONFIG_PATH,
    input_path=paths.source,
    workdir=paths.workdir,
    result_path=paths.result_json,
)

# Inspect or edit paths.result_json here before rendering.
rendered = render_streaming_artifacts(
    input_path=paths.source,
    result_path=paths.result_json,
    redacted_path=paths.result_video,
    config_overrides={"render.redaction.method": "mosaic"},
)
```

For a single analyze-and-render call:

```python
from insightface.app.privateframe import default_output_paths, run_streaming_pipeline
from insightface.app.privateframe.base_config import DEFAULT_CONFIG_PATH

paths = default_output_paths("/data/video.mp4", "/data/python_selected")

result = run_streaming_pipeline(
    config_path=DEFAULT_CONFIG_PATH,
    input_path=paths.source,
    workdir=paths.workdir,
    result_path=paths.result_json,
    redacted_path=paths.result_video,
    debug_path=None,
    config_overrides={
        "recognition.mode": "exempt",
        "recognition.reference_dir": "/data/reference_photos",
    },
)
```

These are alternative workflows; choose distinct output locations when running
both. API callers manage output replacement and concurrency themselves; the
CLI's `--overwrite` check and output locking are not API arguments.
`config_overrides` is a mapping of dotted field names to Python values. Pass a
custom YAML path as `config_path` when needed. Both analysis and rendering also
accept `progress(current, total, stage)` and `is_cancelled()` callbacks. A
cancellation request raises `InterruptedError`.

## CLI discovery, dry runs, and automation

```bash
insightface-privateframe --version
insightface-privateframe describe
insightface-privateframe doctor \
  --input /data/video.mp4 --output-dir /data/output
```

`describe` returns a machine-readable command and configuration contract:
`discovery` explains command selection, `recommended_workflows` gives argument
templates, `primary_io` identifies files, and `config.groups` plus
`config.dotted_options` describe common controls and their tradeoffs.
`config.full_reference.path` locates the complete installed configuration
reference. Discovery uses `contract_schema_version: 2`.

For immediate face redaction, select `process`. Run the intended command with
`--dry-run`, inspect both `ok` and `ready` plus `plan.artifacts`, then run it
without that flag:

```bash
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/output \
  --progress jsonl --dry-run
```

`doctor` and dry runs check environment or job readiness without downloading
models, running inference, or creating output files/directories. Codec checks
can open an in-memory encoder or encode a synthetic frame to a null sink.
Reference-photo suitability is checked only during execution, so a successful
dry run does not establish that photos contain usable faces.

`analyze`, `process`, `render`, `describe`, and `doctor` each write one compact
status JSON object to standard output. Successful execution uses
`status_schema_version: 1`; read final output paths from `artifacts`:

```json
{"status_schema_version":1,"ok":true,"command":"analyze","artifacts":{"result_json":"/data/output/video_privateframe.json","result_video":null},"runtime":{"provider":"CPUExecutionProvider"},"timings":{"total_seconds":8.2},"summary":{"frame_count":300,"face_tracks":3,"face_regions":615}}
```

`timings.total_seconds` covers reported analysis/render stages, excluding
process startup and shutdown. Failures return a structured `error` containing
`code`, `stage`, `type`, `message`, `retryable`, and `hints`. Ctrl-C produces a
`cancelled` status and exit code 130. The full exit-code contract is available
through `describe`.

Progress and diagnostics go to standard error. Execution commands accept
`--progress auto|text|jsonl|none`; `auto` uses text on a terminal and JSONL
otherwise. With `jsonl`, application progress and reference-photo diagnostics
are separate JSONL records; photo logs have `event: "log"`,
`log_schema_version: 1`, and `stage: "recognition"`. Third-party runtime messages
may still be plain text, so do not assume every stderr line is JSON. `none`
suppresses progress but retains diagnostics. Stdout stays the final status JSON.

Concurrent CLI writers targeting the same output or work directory are
serialized by locks. An `output_busy` error can be retried after the active job
finishes. Use separate output directories for independent jobs, and explicitly
add `--overwrite` when replacing an existing result.

## Quality, privacy, and licensing

Face detection, tracking, and reference matching can miss faces or make incorrect
matches, especially with brief appearances, occlusion, small faces, or poor
image quality. Inspect the rendered video before sharing it, including cuts and
frames between detector samples. Blur and mosaic cover detected face regions;
they do not hide voices, names, bodies, clothing, or other identifying context.
They do not guarantee anonymity.

The InsightFace code is MIT-licensed. The pretrained models supplied by the
project, including automatically downloaded models, are available for
**non-commercial research purposes only**. See the
[Python package license notice](../../../README.md#license) when choosing models
for your use case.
