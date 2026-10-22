# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      :

import os
import os.path as osp
import glob
import hashlib
import logging
import platform
import warnings
from collections.abc import Mapping
import numpy as np
import onnx
import onnxruntime
from google.protobuf.message import DecodeError
from .arcface_onnx import *
from .face_verifier import FaceVerifier
from .retinaface import *
from .scrfd import *
from .landmark import *
from .attribute import Attribute
from .inswapper import INSwapper
from .onnxruntime_utils import (
    get_default_providers,
    preload_cuda_libraries,
)
from .coreml_cache import copy_session_options, create_coreml_session
from .package_manifest import (
    DETECTION_TASK,
    EMBEDDED_PREPROCESSING,
    MODEL_PACKAGE_TASKS,
    RECOGNITION_TASK,
    ModelTaskDescriptor,
    has_model_package_manifest,
    load_model_package,
    normalize_preprocessing,
    verify_model_artifact,
)
from ..utils import download_onnx

__all__ = ["get_model"]

logger = logging.getLogger(__name__)

_UNSET = object()
_COREML_PROVIDER = "CoreMLExecutionProvider"
_DEFAULT_COREML_DETECTOR_INPUT_SIZE = (640, 640)


class PickableInferenceSession(onnxruntime.InferenceSession):
    # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, **kwargs):
        if kwargs.get("providers") is None:
            kwargs["providers"] = get_default_providers()
        preload_cuda_libraries(kwargs["providers"])
        super().__init__(model_path, **kwargs)
        self.model_path = model_path

    def __getstate__(self):
        providers = list(self.get_providers())
        provider_options = self.get_provider_options()
        values = {
            "model_path": self.model_path,
            "providers": providers,
            "provider_options": [
                dict(provider_options.get(provider, {})) for provider in providers
            ],
        }
        dimension_overrides = getattr(self, "coreml_dimension_overrides", None)
        if dimension_overrides:
            values["coreml_dimension_overrides"] = dict(dimension_overrides)
        return values

    def __setstate__(self, values):
        model_path = values["model_path"]
        kwargs = {
            "providers": values.get("providers"),
            "provider_options": values.get("provider_options"),
        }
        dimension_overrides = values.get("coreml_dimension_overrides")
        if dimension_overrides:
            kwargs["sess_options"] = copy_session_options(
                dimension_overrides=dimension_overrides,
            )
        self.__init__(model_path, **kwargs)
        if dimension_overrides:
            self.coreml_dimension_overrides = dict(dimension_overrides)


class ModelRouter:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file

    def get_model(self, **kwargs):
        resolution_session_factory = kwargs.pop("resolution_session_factory", None)
        static_shape_sessions = kwargs.pop(
            "static_shape_sessions",
            _UNSET,
        )
        coreml_detector_input_size = kwargs.pop("_coreml_detector_input_size", None)
        coreml_cache_root = kwargs.pop("_coreml_cache_root", None)
        providers = kwargs.get("providers") or ()
        coreml_primary = bool(
            providers and _provider_name(providers[0]) == _COREML_PROVIDER
        )
        if coreml_primary and _onnx_is_detection_model(self.onnx_file):
            session_kwargs = dict(kwargs)
            session_kwargs["_coreml_detector_input_size"] = coreml_detector_input_size
            session_kwargs["_coreml_cache_root"] = coreml_cache_root
            if static_shape_sessions is not _UNSET:
                session_kwargs["static_shape_sessions"] = static_shape_sessions
            session = _managed_detection_session(
                self.onnx_file,
                session_kwargs,
                model_sha256=_model_file_sha256(self.onnx_file),
            )
        else:
            session = PickableInferenceSession(self.onnx_file, **kwargs)
        logger.debug(
            "Applied providers: %s, with options: %s",
            session._providers,
            session._provider_options,
        )
        inputs = session.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()

        if len(outputs) >= 5:
            detector_kwargs = {}
            if resolution_session_factory is not None:
                detector_kwargs["resolution_session_factory"] = (
                    resolution_session_factory
                )
            if static_shape_sessions is not _UNSET:
                detector_kwargs["static_shape_sessions"] = static_shape_sessions
            return SCRFD(
                model_file=self.onnx_file,
                session=session,
                **detector_kwargs,
            )
        elif input_shape[2] == 192 and input_shape[3] == 192:
            return Landmark(model_file=self.onnx_file, session=session)
        elif input_shape[2] == 96 and input_shape[3] == 96:
            return Attribute(model_file=self.onnx_file, session=session)
        elif len(inputs) == 2 and input_shape[2] == 128 and input_shape[3] == 128:
            return INSwapper(model_file=self.onnx_file, session=session)
        elif (
            input_shape[2] == input_shape[3]
            and input_shape[2] >= 112
            and input_shape[2] % 16 == 0
        ):
            return ArcFaceONNX(model_file=self.onnx_file, session=session)
        else:
            # raise RuntimeError('error on model routing')
            return None


def find_onnx_file(dir_path):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob(osp.join(dir_path, "*.onnx"))
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]


def get_default_provider_options():
    return None


def _manifest_directory(name, model_root):
    """Return a manifest directory without changing legacy path lookup."""

    text = os.path.expanduser(str(name))
    direct = osp.abspath(text)
    if osp.isdir(direct) and has_model_package_manifest(direct):
        return direct
    candidate = osp.join(model_root, text)
    if osp.isdir(candidate) and has_model_package_manifest(candidate):
        return candidate
    if text.endswith(".onnx"):
        parent = osp.dirname(osp.abspath(text))
        while parent and parent != osp.dirname(parent):
            if has_model_package_manifest(parent):
                return parent
            parent = osp.dirname(parent)
    return None


def _session_kwargs(kwargs):
    providers = kwargs.get("providers")
    if providers is None:
        providers = get_default_providers()
    values = {
        "providers": providers,
        "provider_options": kwargs.get(
            "provider_options", get_default_provider_options()
        ),
    }
    if kwargs.get("sess_options") is not None:
        values["sess_options"] = kwargs["sess_options"]
    return values


def _provider_name(provider):
    if isinstance(provider, (list, tuple)) and provider:
        return str(provider[0])
    return str(provider)


def _provider_names_and_options(providers, provider_options):
    """Normalize ORT's two accepted provider option representations."""

    providers = list(providers or ())
    names = [_provider_name(provider) for provider in providers]
    embedded = [
        (
            dict(provider[1])
            if isinstance(provider, (list, tuple))
            and len(provider) > 1
            and isinstance(provider[1], Mapping)
            else {}
        )
        for provider in providers
    ]
    if provider_options is None:
        return names, embedded
    if isinstance(provider_options, Mapping):
        if all(
            name in provider_options and isinstance(provider_options[name], Mapping)
            for name in names
        ):
            return names, [dict(provider_options[name]) for name in names]
        if len(names) == 1:
            return names, [dict(provider_options)]
        raise TypeError("provider_options mappings must be keyed by provider name")
    values = [dict(value or {}) for value in provider_options]
    if len(values) > len(names):
        raise ValueError("provider_options has more entries than providers")
    values.extend({} for _provider in range(len(names) - len(values)))
    return names, values


def _coreml_detection_session_values(values, *, static_shapes):
    """Return isolated provider options for one CoreML detector Session."""

    providers, options = _provider_names_and_options(
        values.get("providers"),
        values.get("provider_options"),
    )
    if not providers or providers[0] != _COREML_PROVIDER:
        return dict(values)
    coreml_options = dict(options[0])
    release = platform.mac_ver()[0]
    try:
        macos_major = int(release.split(".", 1)[0])
    except (TypeError, ValueError):
        macos_major = 0
    coreml_options.setdefault(
        "ModelFormat",
        "MLProgram" if macos_major >= 12 else "NeuralNetwork",
    )
    coreml_options.setdefault("EnableOnSubgraphs", "0")
    if static_shapes:
        coreml_options["RequireStaticInputShapes"] = "1"
    else:
        # Dynamic SCRFD + ALL can terminate the process inside CoreML rather
        # than raising a Python exception. The opt-out compatibility path must
        # therefore avoid ALL before constructing the Session.
        coreml_options["MLComputeUnits"] = "CPUAndGPU"
        coreml_options["RequireStaticInputShapes"] = "0"
    options[0] = coreml_options
    result = dict(values)
    result["providers"] = providers
    result["provider_options"] = options
    return result


def _normalize_detector_input_size(value):
    if value is None:
        return _DEFAULT_COREML_DETECTOR_INPUT_SIZE
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("_coreml_detector_input_size must contain width and height")
    width, height = (int(value[0]), int(value[1]))
    if width <= 0 or height <= 0:
        raise ValueError("CoreML detector dimensions must be positive")
    return width, height


def _model_file_sha256(model_file):
    digest = hashlib.sha256()
    with open(model_file, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _onnx_is_detection_model(model_file):
    """Pre-route SCRFD without first constructing a possibly dynamic Session."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                model = onnx.load_model(
                    str(model_file),
                    load_external_data=False,
                )
            except TypeError:
                model = onnx.load_model(str(model_file))
    except (DecodeError, OSError, TypeError, ValueError):
        return False
    # This intentionally mirrors ModelRouter's long-standing runtime rule.
    return len(model.graph.output) >= 5


def _onnx_input_contracts(model_file, dimension_overrides=None):
    """Read stable input metadata before a managed CoreML Session exists."""

    try:
        model = onnx.load_model(str(model_file), load_external_data=False)
    except TypeError:
        model = onnx.load_model(str(model_file))
    overrides = {
        str(name): int(value) for name, value in dict(dimension_overrides or {}).items()
    }
    contracts = []
    for value in model.graph.input:
        dimensions = []
        for dimension in value.type.tensor_type.shape.dim:
            size = int(dimension.dim_value)
            symbol = str(dimension.dim_param)
            if size > 0:
                dimensions.append(size)
            elif symbol in overrides:
                dimensions.append(overrides[symbol])
            else:
                dimensions.append("*")
        try:
            dtype = str(
                np.dtype(
                    onnx.helper.tensor_dtype_to_np_dtype(
                        value.type.tensor_type.elem_type
                    )
                )
            )
        except (TypeError, ValueError):
            dtype = onnx.helper.tensor_dtype_to_string(value.type.tensor_type.elem_type)
        contracts.append(
            {
                "name": value.name,
                "dtype": dtype,
                "shape": dimensions,
            }
        )
    if not contracts:
        raise RuntimeError(f"ONNX model has no graph inputs: {model_file}")
    return contracts


def _detector_dimension_overrides(model_file, input_size):
    """Resolve SCRFD's symbolic NCHW height/width dimensions by name.

    Older buffalo packages use the same ``?`` symbol for both spatial axes,
    while newer packages normally use separate ``height`` and ``width``
    names. A shared symbol can only represent a square static Session.
    """

    width, height = (int(input_size[0]), int(input_size[1]))
    try:
        try:
            model = onnx.load_model(
                str(model_file),
                load_external_data=False,
            )
        except TypeError:
            model = onnx.load_model(str(model_file))
    except (DecodeError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"failed to inspect detector input dimensions: {model_file}"
        ) from error
    if not model.graph.input:
        raise RuntimeError("detector ONNX graph has no input")
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    if len(dimensions) != 4:
        raise RuntimeError("detector input must use NCHW rank 4")

    requested = (height, width)
    spatial = dimensions[2:4]
    overrides = {}
    for axis, (dimension, value) in enumerate(zip(spatial, requested)):
        fixed = int(dimension.dim_value)
        if fixed > 0:
            if fixed != value:
                axis_name = "height" if axis == 0 else "width"
                raise ValueError(
                    f"detector has fixed {axis_name} {fixed}, " f"received {value}"
                )
            continue
        symbol = str(dimension.dim_param)
        if not symbol:
            raise RuntimeError(
                "detector spatial dimensions must have names before they "
                "can be fixed for CoreML"
            )
        previous = overrides.get(symbol)
        if previous is not None and previous != value:
            raise ValueError(
                "detector height and width share the symbolic dimension "
                f"{symbol!r}; CoreML therefore requires a square input size"
            )
        overrides[symbol] = value
    return overrides


def _managed_manifest_session(model_file, task, kwargs):
    """Create one manifest task Session without changing public APIs."""

    descriptor = kwargs.get("model_descriptor")
    values = _session_kwargs(kwargs)
    providers = values["providers"]
    coreml_primary = bool(
        providers and _provider_name(providers[0]) == _COREML_PROVIDER
    )
    if (
        task == DETECTION_TASK
        and coreml_primary
        and (
            isinstance(descriptor, ModelTaskDescriptor)
            or _onnx_is_detection_model(model_file)
        )
    ):
        return _managed_detection_session(
            model_file,
            kwargs,
            model_sha256=_model_file_sha256(model_file),
        )

    if not coreml_primary:
        return PickableInferenceSession(model_file, **values)

    if not isinstance(descriptor, ModelTaskDescriptor):
        return PickableInferenceSession(model_file, **values)

    provider_names, provider_options = _provider_names_and_options(
        providers,
        values.get("provider_options"),
    )
    contracts = _onnx_input_contracts(model_file)
    source_options = values.get("sess_options")
    result = create_coreml_session(
        PickableInferenceSession,
        model_file,
        providers=provider_names,
        provider_options=provider_options,
        model_sha256=_model_file_sha256(model_file),
        task=task,
        graph_variant="onnx_path_v1",
        input_contracts=contracts,
        sess_options_factory=lambda: copy_session_options(
            source_options,
        ),
        warmup=True,
        cache_root=kwargs.get("_coreml_cache_root"),
    )
    return _annotate_coreml_session(result)


def _annotate_coreml_session(result, dimension_overrides=None):
    session = result.session
    session.coreml_compute_units = result.compute_units
    session.coreml_cache_directory = (
        str(result.cache_directory) if result.cache_directory is not None else None
    )
    session.coreml_cache_hit = result.cache_hit
    if dimension_overrides:
        session.coreml_dimension_overrides = dict(dimension_overrides)
    return session


def _managed_detection_session(model_file, kwargs, *, model_sha256):
    """Build SCRFD's main CoreML Session without a dynamic ALL attempt."""

    values = _session_kwargs(kwargs)
    providers = values["providers"]
    if not providers or _provider_name(providers[0]) != _COREML_PROVIDER:
        return PickableInferenceSession(model_file, **values)

    static_shape_sessions = kwargs.get("static_shape_sessions", True)
    if not isinstance(static_shape_sessions, bool):
        raise TypeError("static_shape_sessions must be boolean")
    if not static_shape_sessions:
        dynamic_values = _coreml_detection_session_values(
            values,
            static_shapes=False,
        )
        return PickableInferenceSession(model_file, **dynamic_values)

    width, height = _normalize_detector_input_size(
        kwargs.get("_coreml_detector_input_size")
    )
    dimension_overrides = _detector_dimension_overrides(
        model_file,
        (width, height),
    )
    values = _coreml_detection_session_values(
        values,
        static_shapes=True,
    )
    contracts = _onnx_input_contracts(model_file, dimension_overrides)
    source_options = values.get("sess_options")
    result = create_coreml_session(
        PickableInferenceSession,
        model_file,
        providers=values["providers"],
        provider_options=values.get("provider_options"),
        model_sha256=model_sha256,
        task=DETECTION_TASK,
        graph_variant="onnx_path_dimension_override_v1",
        input_contracts=contracts,
        sess_options_factory=lambda: copy_session_options(
            source_options,
            dimension_overrides,
        ),
        warmup=True,
        cache_root=kwargs.get("_coreml_cache_root"),
    )
    return _annotate_coreml_session(result, dimension_overrides)


def _configure_image_preprocessing(
    model,
    session,
    task,
    preprocessing,
    input_mean,
    input_std,
):
    get_inputs = getattr(session, "get_inputs", None)
    input_type = str(get_inputs()[0].type) if callable(get_inputs) else "tensor(float)"
    if input_type not in {"tensor(float)", "tensor(uint8)"}:
        raise RuntimeError(
            f"{task} preprocessing requires a float32 or uint8 model input; "
            f"received {input_type}"
        )
    if input_type == "tensor(uint8)" and preprocessing != EMBEDDED_PREPROCESSING:
        raise RuntimeError(
            f"{task} mean/std preprocessing requires tensor(float) input"
        )
    model.preprocessing = preprocessing
    model.input_mean = input_mean
    model.input_std = input_std
    model.input_type = input_type
    model.input_dtype = np.uint8 if input_type == "tensor(uint8)" else np.float32
    return model


def _explicit_model(model_file, task, metadata, **kwargs):
    if task not in MODEL_PACKAGE_TASKS:
        raise ValueError("model_task must be detection, verification, or recognition")
    metadata = dict(metadata or {})
    if "preprocessing" not in metadata:
        raise ValueError(f"{task}.preprocessing is required")
    preprocessing = normalize_preprocessing(
        metadata.get("preprocessing"),
        f"{task}.preprocessing",
    )
    if preprocessing == EMBEDDED_PREPROCESSING:
        input_mean = 0.0
        input_std = 1.0
    else:
        input_mean = float(preprocessing["mean"])
        input_std = float(preprocessing["std"])
    session = _managed_manifest_session(model_file, task, kwargs)
    if task == DETECTION_TASK:
        detector_kwargs = {}
        resolution_session_factory = kwargs.get("resolution_session_factory")
        if resolution_session_factory is not None:
            detector_kwargs["resolution_session_factory"] = resolution_session_factory
        if "static_shape_sessions" in kwargs:
            detector_kwargs["static_shape_sessions"] = kwargs["static_shape_sessions"]
        model = SCRFD(
            model_file=model_file,
            session=session,
            **detector_kwargs,
        )
        return _configure_image_preprocessing(
            model,
            session,
            task,
            preprocessing,
            input_mean,
            input_std,
        )
    if task == RECOGNITION_TASK:
        model = ArcFaceONNX(model_file=model_file, session=session)
        return _configure_image_preprocessing(
            model,
            session,
            task,
            preprocessing,
            input_mean,
            input_std,
        )
    return FaceVerifier(
        model_file=model_file,
        session=session,
        expansion=float(metadata.get("expansion", 1.3)),
        preprocessing=preprocessing,
    )


def get_model(name, **kwargs):
    root = kwargs.get("root", "~/.insightface")
    root = os.path.expanduser(root)
    model_root = osp.join(root, "models")
    allow_download = kwargs.get("download", False)
    download_zip = kwargs.get("download_zip", False)
    model_task = kwargs.get("model_task")
    model_metadata = kwargs.get("model_metadata")
    model_descriptor = kwargs.get("model_descriptor")
    name = str(name)

    if model_descriptor is not None:
        if not isinstance(model_descriptor, ModelTaskDescriptor):
            raise TypeError("model_descriptor must be a ModelTaskDescriptor")
        descriptor_task = model_descriptor.task
        if model_task is None:
            model_task = descriptor_task
        if model_task != descriptor_task:
            raise ValueError(
                f"model_task {model_task} does not match descriptor task "
                f"{descriptor_task}"
            )
        model_file = str(verify_model_artifact(model_descriptor))
        return _explicit_model(
            model_file,
            model_task,
            model_descriptor.metadata,
            **kwargs,
        )

    # A direct ONNX path without an explicit task is the legacy API.  Even when
    # the file happens to live beside a manifest, it must retain ModelRouter's
    # historical shape-based behavior.  Package discovery is opt-in through a
    # package directory/name or ``model_task``.
    manifest_dir = (
        None
        if name.endswith(".onnx") and model_task is None
        else _manifest_directory(name, model_root)
    )
    if manifest_dir is not None:
        package = load_model_package(manifest_dir)
        if model_task is None:
            raise ValueError(
                "manifest-backed model packages require an explicit model_task"
            )
        descriptor = package.task(model_task)
        if name.endswith(".onnx"):
            requested = osp.realpath(osp.abspath(os.path.expanduser(name)))
            if requested != str(descriptor.path):
                raise ValueError(
                    f"model file {requested} is not the manifest {model_task} task"
                )
        model_file = str(verify_model_artifact(descriptor))
        explicit_kwargs = dict(kwargs)
        explicit_kwargs["model_descriptor"] = descriptor
        return _explicit_model(
            model_file,
            model_task,
            descriptor.metadata,
            **explicit_kwargs,
        )

    if not name.endswith(".onnx"):
        model_dir = os.path.join(model_root, name)
        model_file = find_onnx_file(model_dir)
        if model_file is None:
            return None
    else:
        model_file = name
    if not osp.exists(model_file) and allow_download:
        model_file = download_onnx(
            "models", model_file, root=root, download_zip=download_zip
        )
    assert osp.exists(model_file), "model_file %s should exist" % model_file
    assert osp.isfile(model_file), "model_file %s should be a file" % model_file
    if model_task is not None:
        return _explicit_model(
            model_file,
            model_task,
            model_metadata,
            **kwargs,
        )
    router = ModelRouter(model_file)
    router_kwargs = _session_kwargs(kwargs)
    resolution_session_factory = kwargs.get("resolution_session_factory")
    if resolution_session_factory is not None:
        router_kwargs["resolution_session_factory"] = resolution_session_factory
    if "static_shape_sessions" in kwargs:
        router_kwargs["static_shape_sessions"] = kwargs["static_shape_sessions"]
    for internal_name in (
        "_coreml_detector_input_size",
        "_coreml_cache_root",
    ):
        if internal_name in kwargs:
            router_kwargs[internal_name] = kwargs[internal_name]
    model = router.get_model(**router_kwargs)
    return model
