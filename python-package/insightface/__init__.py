# coding: utf-8
# pylint: disable=wrong-import-position
"""InsightFace: A Face Analysis Toolkit."""
from __future__ import absolute_import
from loguru import logger
import os

try:
    #import mxnet as mx
    import onnxruntime
except ImportError:
    raise ImportError(
        "Unable to import dependency onnxruntime. "
    )

if os.getenv("DEBUG") is None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

__version__ = '0.7.3'

from . import model_zoo
from . import utils
from . import app
from . import data
from . import thirdparty

