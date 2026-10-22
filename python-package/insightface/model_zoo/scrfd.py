# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

from __future__ import division

import datetime
import hashlib
import os
import os.path as osp
import platform
import threading
from functools import lru_cache

import cv2
import numpy as np
import onnx
import onnxruntime
from google.protobuf.message import DecodeError

from .coreml_cache import copy_session_options, create_coreml_session
from .onnxruntime_utils import get_default_providers

DEFAULT_DET_SIZES = [(128, 128), (640, 640)]

_COREML_PROVIDER = 'CoreMLExecutionProvider'
_UNKNOWN_INPUT_SIZE = object()


def _fixed_input_size(input_shape):
    """Return ``(width, height)`` when an NCHW shape fixes both axes."""

    if input_shape is None or len(input_shape) < 4:
        return None
    height, width = input_shape[2:4]
    if not all(
        isinstance(value, (int, np.integer)) and int(value) > 0
        for value in (height, width)
    ):
        return None
    return int(width), int(height)


def _native_model_input_size(model_file, input_name=None):
    """Inspect the ONNX graph, independently of Session dimension overrides.

    A provider may turn a dynamic graph input into a fixed effective Session
    input (CoreML's free-dimension overrides do this).  The graph remains the
    source of truth for deciding whether resolution-specific Sessions may be
    created.  ``_UNKNOWN_INPUT_SIZE`` lets callers fall back to Session
    metadata when an injected/fake Session has no readable model file.
    """

    if model_file is None or not osp.isfile(model_file):
        return _UNKNOWN_INPUT_SIZE
    try:
        try:
            model = onnx.load_model(
                str(model_file),
                load_external_data=False,
            )
        except TypeError:
            model = onnx.load_model(str(model_file))
        if input_name is None:
            value = model.graph.input[0] if model.graph.input else None
        else:
            graph_inputs = {
                value.name: value for value in model.graph.input
            }
            value = graph_inputs.get(input_name)
        if value is None:
            return _UNKNOWN_INPUT_SIZE
        dimensions = value.type.tensor_type.shape.dim
        if len(dimensions) < 4:
            return _UNKNOWN_INPUT_SIZE
        spatial = []
        for dimension in dimensions[2:4]:
            size = int(dimension.dim_value)
            if size <= 0:
                return None
            spatial.append(size)
        return spatial[1], spatial[0]
    except (AttributeError, DecodeError, IndexError, OSError, TypeError, ValueError):
        # Session metadata retains the historical behavior for invalid paths,
        # custom Session implementations, and older ONNX protobuf versions.
        return _UNKNOWN_INPUT_SIZE


def _session_providers(session):
    getter = getattr(session, 'get_providers', None)
    if callable(getter):
        return tuple(str(value) for value in getter())
    values = getattr(session, '_providers', ())
    return tuple(str(value) for value in (values or ()))


def _session_provider_options(session, providers):
    getter = getattr(session, 'get_provider_options', None)
    if callable(getter):
        options = getter()
        if isinstance(options, dict):
            return tuple(dict(options.get(provider, {})) for provider in providers)
        if options is not None:
            return tuple(dict(value) for value in options)
    options = getattr(session, '_provider_options', None)
    if options is None:
        return None
    if isinstance(options, dict):
        return tuple(dict(options.get(provider, {})) for provider in providers)
    return tuple(dict(value) for value in options)


def _freeze_provider_value(value):
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _freeze_provider_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_provider_value(item) for item in value)
    return value


def _session_provider_signature(session):
    providers = _session_providers(session)
    options = _session_provider_options(session, providers)
    return providers, _freeze_provider_value(options)


def _fresh_coreml_session_options(reference_session):
    """Copy safe scalar settings without reusing CoreML SessionOptions state.

    ONNX Runtime may crash when one SessionOptions instance is first used to
    build a dynamic CoreML Session and then reused for a static model. Custom
    ops, external initializers, optimized-model output paths, and private
    config entries are not enumerable or safe to share; callers that depend on
    those must provide ``resolution_session_factory``.
    """

    return copy_session_options(reference_session)


def _coreml_static_provider_options(options):
    """Return safe fixed-shape CoreML defaults plus caller-visible options."""

    values = dict(options)
    release = platform.mac_ver()[0]
    try:
        major = int(release.split('.', 1)[0])
    except (TypeError, ValueError):
        major = 0
    values.setdefault(
        'ModelFormat',
        'MLProgram' if major >= 12 else 'NeuralNetwork',
    )
    values.setdefault('MLComputeUnits', 'ALL')
    values.setdefault('EnableOnSubgraphs', '0')
    values['RequireStaticInputShapes'] = '1'
    # A reference Session's directory describes a different input contract.
    # create_coreml_session assigns a signature-specific directory below.
    values.pop('ModelCacheDirectory', None)
    return values


@lru_cache(maxsize=128)
def _model_file_sha256_cached(path, size, mtime_ns):
    """Hash one immutable-looking file identity used by derived Sessions."""

    del size, mtime_ns  # They intentionally participate in the cache key.
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _model_file_sha256(model_file):
    path = osp.realpath(osp.abspath(osp.expanduser(str(model_file))))
    stat = os.stat(path)
    mtime_ns = getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1_000_000_000))
    return _model_file_sha256_cached(path, int(stat.st_size), int(mtime_ns))


def _set_dimension_value(dimension, value):
    dimension.ClearField('dim_param')
    dimension.dim_value = int(value)


def _static_scrfd_model(model_file, input_size, input_name):
    """Return SCRFD model bytes with exact input and output dimensions."""

    width, height = (int(input_size[0]), int(input_size[1]))
    try:
        model = onnx.load_model(str(model_file), load_external_data=True)
        # Serialized in-memory models have no directory from which ORT can
        # resolve external tensors, so embed them before serialization.
        onnx.external_data_helper.convert_model_from_external_data(model)
    except (DecodeError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f'failed to load SCRFD model for a static Session: {model_file}'
        ) from exc

    graph_inputs = {value.name: value for value in model.graph.input}
    input_value = graph_inputs.get(input_name)
    if input_value is None:
        raise RuntimeError(
            f'SCRFD ONNX graph has no input named {input_name!r}'
        )
    input_dimensions = input_value.type.tensor_type.shape.dim
    if len(input_dimensions) != 4:
        raise RuntimeError('SCRFD input must use NCHW rank 4')
    _set_dimension_value(input_dimensions[2], height)
    _set_dimension_value(input_dimensions[3], width)

    output_count = len(model.graph.output)
    if output_count in (6, 9):
        fmc = 3
        strides = (8, 16, 32)
        anchors = 2
    elif output_count in (10, 15):
        fmc = 5
        strides = (8, 16, 32, 64, 128)
        anchors = 1
    else:
        raise RuntimeError(
            'SCRFD static output shaping requires 6, 9, 10, or '
            f'15 outputs; received {output_count}'
        )
    feature_widths = (1, 4, 10)
    for index, output in enumerate(model.graph.output):
        dimensions = output.type.tensor_type.shape.dim
        if len(dimensions) not in (2, 3):
            raise RuntimeError(
                'SCRFD outputs must have rank 2 or 3; '
                f'{output.name!r} has rank {len(dimensions)}'
            )
        if len(dimensions) == 3:
            _set_dimension_value(dimensions[0], 1)
        candidate_axis = len(dimensions) - 2
        stride = strides[index % fmc]
        candidates = (height // stride) * (width // stride) * anchors
        _set_dimension_value(dimensions[candidate_axis], candidates)
        component = index // fmc
        _set_dimension_value(dimensions[-1], feature_widths[component])
    return model.SerializeToString()


def _default_resolution_session_factory(
    model_file,
    input_size,
    reference_session,
):
    """Create an independent Session with safely reusable reference settings."""

    if model_file is None:
        # A plain externally injected ORT Session does not expose the model
        # source needed to construct a clone.  Preserve SCRFD's historical
        # session-only API by sharing that Session across sizes; callers that
        # supply model_file still receive one independent Session per size.
        return reference_session
    providers = _session_providers(reference_session)
    provider_options = _session_provider_options(reference_session, providers)
    coreml_primary = bool(providers and providers[0] == _COREML_PROVIDER)
    kwargs = {}
    options_getter = getattr(reference_session, 'get_session_options', None)
    if not coreml_primary and callable(options_getter):
        kwargs['sess_options'] = options_getter()
    if providers:
        kwargs['providers'] = list(providers)
        if provider_options is not None or coreml_primary:
            normalized_options = (
                [dict(value) for value in provider_options]
                if provider_options is not None
                else [{} for _provider in providers]
            )
            while len(normalized_options) < len(providers):
                normalized_options.append({})
            if coreml_primary:
                normalized_options[0] = _coreml_static_provider_options(
                    normalized_options[0]
                )
            kwargs['provider_options'] = normalized_options
    input_getter = getattr(reference_session, 'get_inputs', None)
    if not callable(input_getter) or not input_getter():
        raise RuntimeError('SCRFD reference Session has no input')
    input_metadata = input_getter()[0]
    input_name = input_metadata.name
    model_source = _static_scrfd_model(
        model_file,
        input_size,
        input_name,
    )
    if coreml_primary:
        width, height = (int(input_size[0]), int(input_size[1]))
        result = create_coreml_session(
            onnxruntime.InferenceSession,
            model_source,
            providers=kwargs['providers'],
            provider_options=kwargs.get('provider_options'),
            model_sha256=_model_file_sha256(model_file),
            task='detection',
            graph_variant='static_scrfd_rewrite_v1',
            input_contracts=(
                {
                    'name': input_name,
                    'dtype': getattr(
                        input_metadata,
                        'type',
                        'tensor(float)',
                    ),
                    'shape': [1, 3, height, width],
                },
            ),
            sess_options_factory=lambda: _fresh_coreml_session_options(
                reference_session
            ),
            warmup=True,
        )
        return result.session
    return onnxruntime.InferenceSession(model_source, **kwargs)


class _ResolutionSessionPool:
    """Thread-safe, lazily populated SCRFD Session routing state."""

    def __init__(
        self,
        main_session,
        model_file,
        session_factory,
        native_static_input_size,
        effective_main_input_size,
        static_shape_sessions,
    ):
        self.main_session = main_session
        self.model_file = model_file
        self.session_factory = session_factory
        self.native_static_input_size = native_static_input_size
        self.effective_main_input_size = effective_main_input_size
        self.static_shape_sessions = bool(static_shape_sessions)
        # A dynamic graph may still have an effectively fixed main Session
        # because the provider was configured with a dimension override.
        self._provider_signature = _session_provider_signature(main_session)
        self._lock = threading.Lock()
        self._reset_derived_locked()

    def _reset_derived_locked(self):
        self._sessions = {}
        if self.native_static_input_size is not None:
            self.main_input_size = tuple(self.native_static_input_size)
        else:
            self.main_input_size = self.effective_main_input_size
        if self.main_input_size is not None:
            self._sessions[tuple(self.main_input_size)] = self.main_session

    def _refresh_provider_locked(self):
        signature = _session_provider_signature(self.main_session)
        if signature != self._provider_signature:
            self._provider_signature = signature
            self._reset_derived_locked()

    def clear_derived(self):
        with self._lock:
            self._provider_signature = _session_provider_signature(
                self.main_session
            )
            self._reset_derived_locked()

    def session_for(self, input_size):
        key = tuple(int(value) for value in input_size)
        if self.native_static_input_size is not None:
            static_key = tuple(self.native_static_input_size)
            if key != static_key:
                raise ValueError(
                    'static SCRFD input only supports resolution '
                    f'{static_key}, received {key}'
                )
        with self._lock:
            self._refresh_provider_locked()
            session = self._sessions.get(key)
            if session is not None:
                return session
            if not self.static_shape_sessions or self.model_file is None:
                # Compatibility mode intentionally routes every resolution to
                # the original dynamic Session.  A session-only SCRFD has no
                # ONNX source from which an exact-shape Session can be built,
                # so it follows the same behavior regardless of the flag.
                self._sessions[key] = self.main_session
                return self.main_session
            session = self.session_factory(
                self.model_file,
                key,
                self.main_session,
            )
            expected = _session_providers(self.main_session)
            actual = _session_providers(session)
            if expected and actual != expected:
                raise RuntimeError(
                    'SCRFD resolution Session activated providers '
                    f'{list(actual)}, expected {list(expected)}'
                )
            input_getter = getattr(session, 'get_inputs', None)
            inputs = input_getter() if callable(input_getter) else ()
            actual_input_size = (
                _fixed_input_size(inputs[0].shape) if inputs else None
            )
            if actual_input_size != key:
                raise RuntimeError(
                    'SCRFD resolution Session must have fixed input size '
                    f'{key}, received {actual_input_size}'
                )
            self._sessions[key] = session
            return session

    def snapshot(self):
        with self._lock:
            self._refresh_provider_locked()
            return dict(self._sessions)

    def __getstate__(self):
        # ORT Sessions other than the main PickableInferenceSession and locks
        # are not pickleable.  Derived Sessions are intentionally lazy again
        # after unpickling.
        with self._lock:
            return {
                'main_session': self.main_session,
                'model_file': self.model_file,
                'session_factory': self.session_factory,
                'native_static_input_size': self.native_static_input_size,
                'effective_main_input_size': self.effective_main_input_size,
                'static_shape_sessions': self.static_shape_sessions,
                'main_input_size': self.main_input_size,
                'main_session_keys': tuple(
                    key
                    for key, session in self._sessions.items()
                    if session is self.main_session
                ),
            }

    def __setstate__(self, values):
        self.__dict__.update(values)
        self._provider_signature = _session_provider_signature(
            self.main_session
        )
        self._lock = threading.Lock()
        input_getter = getattr(self.main_session, 'get_inputs', None)
        inputs = input_getter() if callable(input_getter) else ()
        self.effective_main_input_size = (
            _fixed_input_size(inputs[0].shape) if inputs else None
        )
        # PickableInferenceSession recreates itself from model_path and may
        # therefore lose a free-dimension override. Never restore a fixed-size
        # alias merely because it belonged to the pre-pickle Session.
        self._reset_derived_locked()
        if not self.static_shape_sessions:
            for key in values.get('main_session_keys', ()):
                self._sessions[tuple(key)] = self.main_session


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    def __init__(
        self,
        model_file=None,
        session=None,
        resolution_session_factory=None,
        static_shape_sessions=True,
    ):
        import onnxruntime
        self.model_file = model_file
        self.session = session
        self.resolution_session_factory = (
            _default_resolution_session_factory
            if resolution_session_factory is None
            else resolution_session_factory
        )
        if not isinstance(static_shape_sessions, bool):
            raise TypeError('static_shape_sessions must be boolean')
        self.static_shape_sessions = static_shape_sessions
        self.taskname = 'detection'
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(
                self.model_file,
                providers=get_default_providers(),
            )
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()
        self._reset_resolution_session_pool()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        #print(input_shape)
        effective_input_size = _fixed_input_size(input_shape)
        native_input_size = _native_model_input_size(
            self.model_file,
            input_cfg.name,
        )
        if native_input_size is _UNKNOWN_INPUT_SIZE:
            native_input_size = effective_input_size
        self.static_input_size = native_input_size
        self._effective_session_input_size = effective_input_size
        self.input_size = self.static_input_size if self.static_input_size is not None else DEFAULT_DET_SIZES[-1]
        self.input_sizes = [self.static_input_size] if self.static_input_size is not None else list(DEFAULT_DET_SIZES)
        self._debug_det_size_printed = False
        #print('image_size:', self.image_size)
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
        #print(self.output_names)
        #assert len(outputs)==10 or len(outputs)==15
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        if len(outputs)==6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs)==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs)==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs)==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def _reset_resolution_session_pool(self):
        self._resolution_session_pool = _ResolutionSessionPool(
            self.session,
            self.model_file,
            self.resolution_session_factory,
            self.static_input_size,
            self._effective_session_input_size,
            self.static_shape_sessions,
        )

    def _current_resolution_session_pool(self):
        pool = getattr(self, '_resolution_session_pool', None)
        if pool is None or pool.main_session is not self.session:
            self._reset_resolution_session_pool()
            pool = self._resolution_session_pool
        return pool

    def _session_for_input_size(self, input_size):
        return self._current_resolution_session_pool().session_for(input_size)

    @property
    def resolution_sessions(self):
        """Snapshot mapping ``(width, height)`` to initialized Sessions."""

        return self._current_resolution_session_pool().snapshot()

    @property
    def resolution_session_input_sizes(self):
        """Initialized resolution keys, ordered for deterministic inspection."""

        return tuple(sorted(self.resolution_sessions))

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])
            self._current_resolution_session_pool().clear_derived()
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.static_input_size is not None:
                self.input_size = self.static_input_size
                self.input_sizes = [self.static_input_size]
            else:
                self.input_sizes = self._normalize_input_sizes(input_size)
                self.input_size = self.input_sizes[-1]
            self._debug_det_size_printed = False

    def _prepare_input_blob(self, img, input_size):
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        return np.ascontiguousarray(
            blob,
            dtype=getattr(self, 'input_dtype', np.float32),
        )

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = self._prepare_input_blob(img, input_size)
        session = self._session_for_input_size(input_size)
        net_outs = session.run(
            self.output_names,
            {self.input_name : blob},
        )

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                #solution-1, c style:
                #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                #for i in range(height):
                #    anchor_centers[i, :, 1] = i
                #for i in range(width):
                #    anchor_centers[:, i, 0] = i

                #solution-2:
                #ax = np.arange(width, dtype=np.float32)
                #ay = np.arange(height, dtype=np.float32)
                #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                #solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                #print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(
        self,
        img,
        input_size=None,
        max_num=0,
        metric='default',
        det_thresh=None,
    ):
        threshold = (
            getattr(self, 'det_thresh', 0.5)
            if det_thresh is None
            else float(det_thresh)
        )
        input_sizes = self._resolve_input_sizes(input_size)
        assert input_sizes
        pre_det_list = []
        kpss_det_list = []
        for input_size in input_sizes:
            pre_det, kpss = self._detect_candidates(
                img,
                input_size,
                threshold,
            )
            if pre_det.shape[0] == 0:
                continue
            pre_det_list.append(pre_det)
            if self.use_kps and kpss is not None:
                kpss_det_list.append(kpss)
        if not pre_det_list:
            det = np.empty((0, 5), dtype=np.float32)
            kpss = np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None
            return det, kpss
        pre_det = np.vstack(pre_det_list).astype(np.float32, copy=False)
        order = np.argsort(-pre_det[:, 4], kind='stable')
        pre_det = pre_det[order, :]
        if self.use_kps and len(kpss_det_list) == len(pre_det_list):
            kpss = np.vstack(kpss_det_list)[order, :, :]
        else:
            kpss = None
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if kpss is not None:
            kpss = kpss[keep, :, :]
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def _detect_candidates(self, img, input_size, threshold):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, threshold)

        if len(scores_list) == 0 or sum(score.size for score in scores_list) == 0:
            kps_shape = (0, 5, 2) if self.use_kps else None
            return np.empty((0, 5), dtype=np.float32), np.empty(kps_shape, dtype=np.float32) if kps_shape else None
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = np.argsort(-scores_ravel, kind='stable')
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
        else:
            kpss = None
        return pre_det, kpss

    def _resolve_input_sizes(self, input_size):
        static_input_size = getattr(self, 'static_input_size', None)
        if static_input_size is not None:
            return [tuple(static_input_size)]
        if input_size is not None:
            return self._normalize_input_sizes(input_size)
        if self.input_sizes:
            return list(self.input_sizes)
        if self.input_size is not None:
            return [self.input_size]
        return list(DEFAULT_DET_SIZES)

    @staticmethod
    def _normalize_input_sizes(input_size):
        if input_size is None:
            return []
        if isinstance(input_size, np.ndarray):
            input_size = input_size.tolist()
        if (
            isinstance(input_size, (list, tuple))
            and len(input_size) > 0
            and isinstance(input_size[0], (list, tuple, np.ndarray))
        ):
            values = input_size
        else:
            values = [input_size]
        sizes = []
        for item in values:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if len(item) != 2:
                raise ValueError('det_size must be a pair or a list of pairs')
            width, height = int(item[0]), int(item[1])
            if width == 0 and height == 0:
                for size in DEFAULT_DET_SIZES:
                    if size not in sizes:
                        sizes.append(size)
                continue
            if width <= 0 or height <= 0:
                raise ValueError('det_size values must be positive')
            size = (width, height)
            if size not in sizes:
                sizes.append(size)
        return sizes

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(-scores, kind='stable')

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

def get_scrfd(name, download=False, root='~/.insightface/models', **kwargs):
    if not download:
        assert os.path.exists(name)
        return SCRFD(name, **kwargs)
    else:
        from .model_store import get_model_file
        _file = get_model_file("scrfd_%s" % name, root=root)
        return SCRFD(_file, **kwargs)


def scrfd_2p5gkps(**kwargs):
    return get_scrfd("2p5gkps", download=True, **kwargs)


if __name__ == '__main__':
    detector = SCRFD(model_file='./det.onnx')
    detector.prepare(-1)
    img_paths = ['tests/data/t1.jpg']
    for img_path in img_paths:
        img = cv2.imread(img_path)

        for _ in range(1):
            ta = datetime.datetime.now()
            #bboxes, kpss = detector.detect(img, 0.5, input_size = (640, 640))
            bboxes, kpss = detector.detect(img, 0.5)
            tb = datetime.datetime.now()
            print('all cost:', (tb-ta).total_seconds()*1000)
        print(img_path, bboxes.shape)
        if kpss is not None:
            print(kpss.shape)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2,score = bbox.astype(int)
            cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(int)
                    cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
        filename = osp.basename(img_path)
        print('output:', filename)
        cv2.imwrite(osp.join('outputs', filename), img)
