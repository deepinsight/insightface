# coding: utf-8
# pylint: disable=wrong-import-position
"""InsightFace: A Face Analysis Toolkit."""
from __future__ import absolute_import

import os

# ORT reads this during native initialization, so set it before importing ORT.
# Older runtimes that do not recognize this variable simply ignore it.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

try:
    #import mxnet as mx
    import onnxruntime
except ModuleNotFoundError as exc:
    if exc.name != "onnxruntime":
        raise
    raise ImportError(
        "InsightFace requires ONNX Runtime for inference, but the "
        "'onnxruntime' module could not be imported. The default InsightFace "
        "installation includes `onnxruntime` for CPU/CoreML; restore it with "
        "`python -m pip install onnxruntime`. For NVIDIA CUDA, install "
        "`onnxruntime-gpu` only after uninstalling `onnxruntime`. Do not keep "
        "both runtime distributions in the same environment."
    ) from exc
except ImportError as exc:
    raise ImportError(
        "The installed ONNX Runtime package could not be imported. Verify "
        "that its Python, operating-system, CUDA, and cuDNN versions are "
        f"compatible. Original error: {exc}"
    ) from exc

__version__ = '2.1'

from . import model_zoo
from . import utils
from . import app
from . import data
from . import thirdparty
