# Installation and runtime

[Back to the Python library](../README.md)

InsightFace 2.1 uses ONNX Runtime for FaceAnalysis, ModelZoo, PersonAnalysis, PrivateFrame, and
the desktop Evaluation Studio. This guide covers installation, execution
provider selection, telemetry settings, and CoreML compilation caches. See
the [model guide](model_zoo.md) for model packages and direct model loading.

## Installation

Python 3.10 or newer is required for the base package and all optional extras.

| Use case | Command |
| --- | --- |
| FaceAnalysis, PersonAnalysis and ModelZoo | `python -m pip install -U insightface` |
| Video face blur/mosaic with the PrivateFrame API and CLI | `python -m pip install -U "insightface[privateframe]"` |
| Evaluation Studio GUI, including PrivateFrame | `python -m pip install -U "insightface[gui]"` |

The base package installs `onnxruntime`. The `privateframe` extra additionally
installs PyAV and PyYAML; the `gui` extra includes those dependencies plus the
Qt desktop application.

### Install from source

Run these commands from the repository root:

```bash
# Base SDK, including PersonAnalysis:
python -m pip install -e ./python-package
# Or include the optional PrivateFrame workflow:
python -m pip install -e "./python-package[privateframe]"
# Or install the desktop application, which includes PrivateFrame:
python -m pip install -e "./python-package[gui]"
```

### Launch the optional applications

```bash
# Desktop GUI
insightface-gui

# PrivateFrame CLI
insightface-privateframe --help
# Equivalent: python -m insightface_privateframe_bootstrap --help
```

See the [Evaluation Studio guide](gui.md) and
[PrivateFrame guide](../insightface/app/privateframe/README.md) for their
workflows and configuration.

### Optional face3d installation

Ordinary inference and GUI use do not require the optional Cython/C++ `face3d`
extension. If your application uses it, install its dependencies and compile it
from a source checkout with a suitable C++ compiler:

```bash
cd python-package
python -m pip install -e ".[face3d]" --no-build-isolation --config-settings editable_mode=compat
python setup.py build_ext --inplace --with-face3d
```

## Automatic provider selection

When callers do not pass an explicit provider list, FaceAnalysis and ModelZoo inspect the
providers reported by the installed ONNX Runtime and select the first
available entry in this order:

```text
CoreMLExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider
```

Only one accelerated provider is selected. CPU is appended as its fallback
when available. For example, a runtime that reports both CoreML and CUDA uses
CoreML + CPU, not CoreML + CUDA + CPU. Explicit `providers=[...]` arguments and
PrivateFrame's explicit `runtime.provider` setting take precedence over this
automatic policy.

PersonAnalysis supports CPU and CUDA. Its Python API defaults to
`CPUExecutionProvider`; pass `providers=["CUDAExecutionProvider"]` to request
CUDA. In the GUI's Person Analysis workflow, **Auto** chooses CUDA when available,
otherwise CPU. Cheetah does not use CoreML. See the
[PersonAnalysis guide](person_analysis.md) for model setup and examples.

Check what the current Python environment can actually use:

```bash
python -c "import insightface; import onnxruntime as ort; print(ort.get_available_providers())"
```

## macOS and CoreML

No separate InsightFace CoreML package is required. CoreML is selected
automatically only when the installed ONNX Runtime reports
`CoreMLExecutionProvider`; otherwise InsightFace uses the next available
provider.

On CoreML, SCRFD uses a fixed 640x640 main session by default and lazily
creates one reusable fixed-shape session for each additional detection
resolution. Auto detection size uses 128x128 and 640x640. For
InsightFace-managed CoreML sessions, compiled artifacts are stored under
`~/.insightface/cache/coreml/v1`, scoped by the model and input signature.
A new signature may take longer on its first compilation and is warmed up
once afterward; valid cache hits are reused without another warmup.

InsightFace first tries all CoreML compute units and falls back to CPU + GPU
when necessary. These cache controls are internal and do not require
additional arguments to `FaceAnalysis` or `model_zoo.get_model()`.

## NVIDIA CUDA

The Python package intentionally has no `gpu` extra. Install InsightFace, then
replace the default runtime with the GPU distribution:

```bash
python -m pip install -U insightface  # or insightface[privateframe] / [gui]
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Do not keep `onnxruntime` and `onnxruntime-gpu` installed together. Installing
or upgrading InsightFace may install its declared `onnxruntime` dependency
again, so repeat the replacement afterward on NVIDIA systems.

## ONNX Runtime telemetry

InsightFace sets `ORT_DISABLE_TELEMETRY=1` before importing ONNX Runtime; no
shell configuration is required. This also overrides an existing value of
`0`. On runtimes that support this switch, it prevents the non-Windows
telemetry uploader from starting. Older runtimes that do not recognize the
variable ignore it, so it does not introduce a newer ONNX Runtime requirement.

Import InsightFace before importing ONNX Runtime elsewhere in your process:
the switch cannot undo telemetry initialization that has already happened.
See [ONNX Runtime's telemetry documentation](https://github.com/microsoft/onnxruntime/blob/v1.29.0/docs/Privacy.md#disabling-telemetry)
for platform-specific behavior.
