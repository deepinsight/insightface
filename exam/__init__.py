# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""NDArray API of MXNet."""

from . import _internal, contrib, linalg, op, random, sparse, utils, image, ndarray
# pylint: disable=wildcard-import, redefined-builtin
try:
    from .gen_op import * # pylint: disable=unused-wildcard-import
except ImportError:
    pass
from . import register
from .op import *
from .ndarray import *
# pylint: enable=wildcard-import
from .utils import load, load_frombuffer, save, zeros, empty, array
from .sparse import _ndarray_cls
from .ndarray import _GRAD_REQ_MAP, _DTYPE_MX_TO_NP, _DTYPE_NP_TO_MX, _new_empty_handle

__all__ = op.__all__ + ndarray.__all__ + utils.__all__ + \
          ['contrib', 'linalg', 'random', 'sparse', 'image']
