# Change Log

Release history for the InsightFace Python Library. See the
[package README](README.md) for installation and quick starts.

## [2.1] - 2026-10-03

### PersonAnalysis

- Add person detection, face recognition and body ReID through `PersonAnalysis`,
  with separate `get()`, `match()` and optional `update()` calls.
- Support `cheetah_s` and `cheetah_l` on CPU and NVIDIA CUDA, with first-use model
  downloads and configurable face and body detection input sizes.
- Add a desktop workflow for local video, webcams and RTSP, with reference-photo
  registration, analysis-rate controls and advanced settings.

See the [PersonAnalysis guide](docs/person_analysis.md) for usage, configuration
and examples, or the [GUI guide](docs/gui.md) for desktop use.

## [2.0] - 2026-09-10

### Liveness update

- Add opt-in RGB liveness through `FaceAnalysis(addons=["liveness"])`, with
  `normal` / `observe` modes and a configurable live-score threshold (default
  `0.8`).
- Return per-face scores and input-rejection guidance. Normal mode retains
  detections while skipping recognition for faces that do not pass.
- Download and verify the separate addon model under
  `<root>/addons/liveness.onnx`. See [usage and result fields](docs/liveness.md).

### PrivateFrame update

- Add local video face detection/tracking with Gaussian blur or mosaic through
  the Python API, `insightface-privateframe` CLI, and Evaluation Studio GUI.
- Add `all`, `blur_only`, and `exempt` policies, with a folder of reference
  photos for selecting people.
- Export editable analysis JSON and render it again without another inference
  pass; provide `analyze`, `process`, and `render` workflows.
- Add configurable analysis sampling (Fast mode, target 15 FPS by default),
  interpolation between scans, encoding/audio controls, and structured CLI
  status and progress.
- Add the [PrivateFrame feature and usage guide](insightface/app/privateframe/README.md).

### Runtime and model updates

- Add manifest-backed `raccoon_s` / `raccoon_l` model packages and make
  `raccoon_s` the default for new GUI configurations.
- Add automatic CoreML/CUDA/CPU provider selection and persistent CoreML
  compilation caches; use reusable fixed-shape SCRFD sessions per resolution.
- Set `ORT_DISABLE_TELEMETRY=1` before importing ONNX Runtime, where supported.
- Add the `privateframe` installation extra and include it in the `gui` extra.

## [1.0.1] - 2026-05-23

### Changed

- Install ``onnxruntime`` by default for CPU and supported macOS CoreML systems.
  NVIDIA CUDA users must replace it with ``onnxruntime-gpu`` after installing
  or upgrading InsightFace; the two runtime distributions must not coexist.
- Remove the PyPI package metadata license classifier field while keeping the README license guidance.
- Move direct `Pillow` and `scikit-learn` requirements to the GUI extra, and `matplotlib` to the optional `face3d` extra.
- Remove unused base dependencies on `easydict` and `prettytable`.

## [1.0] - 2026-05-23

### Added

- Add **InsightFace Evaluation Studio**, a cross-platform PySide6 desktop GUI for local face recognition, album grouping, enterprise evaluation/report export, and face swap trials.
- Add GUI launch commands: ``insightface-gui``, ``insightface-eval-studio``, ``insightface-desktop``, and ``python -m insightface.gui``.

### Changed

- ``FaceAnalysis.prepare()`` now defaults ``ctx_id`` to 0 and uses Auto detection size, running SCRFD at both 128x128 and 640x640 before unified NMS.
- Route detection models loaded by ``model_zoo.get_model()`` through ``SCRFD`` by default.
- The optional ``face3d`` Cython/C++ extension is no longer built by default; use ``--with-face3d`` or ``INSIGHTFACE_WITH_FACE3D=1`` to opt in.

## [0.7.1] - 2022-12-14

### Changed

- Change model downloading provider to cloudfront.

## [0.7] - 2022-11-28

### Added

- Add face swapping model and example.

### Changed

- Set default ORT provider to CUDA and CPU.

## [0.6] - 2022-01-29

### Added

- Add pose estimation in face-analysis app.

### Changed

- Change model automated downloading url, to ucloud.
