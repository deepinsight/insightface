# coding: utf-8
# pylint: disable=wrong-import-position
"""InsightFace: A Face Analysis Toolkit."""
from __future__ import absolute_import

try:
    #import mxnet as mx
    import onnxruntime
except ImportError:
    raise ImportError(
        "Unable to import dependency onnxruntime. "
    )

__version__ = '0.4'

from . import model_zoo
from . import utils
from . import app
from . import data
from . import thirdparty

