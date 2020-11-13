# coding: utf-8
# pylint: disable=wrong-import-position
"""InsightFace: A Face Analysis Toolkit."""
from __future__ import absolute_import

# mxnet version check
#mx_version = '1.4.0'
try:
    import mxnet as mx
    #from distutils.version import LooseVersion
    #if LooseVersion(mx.__version__) < LooseVersion(mx_version):
    #    msg = (
    #        "Legacy mxnet-mkl=={} detected, some new modules may not work properly. "
    #        "mxnet-mkl>={} is required. You can use pip to upgrade mxnet "
    #        "`pip install mxnet-mkl --pre --upgrade` "
    #        "or `pip install mxnet-cu90mkl --pre --upgrade`").format(mx.__version__, mx_version)
    #    raise ImportError(msg)
except ImportError:
    raise ImportError(
        "Unable to import dependency mxnet. "
        "A quick tip is to install via `pip install mxnet-mkl/mxnet-cu90mkl --pre`. "
    )

__version__ = '0.1.5'

from . import model_zoo
from . import utils
from . import app
