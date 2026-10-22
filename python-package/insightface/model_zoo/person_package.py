"""V2 PersonAnalysis packages, with local validation and first-use downloads."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

from .package_manifest import (
    EMBEDDED_PREPROCESSING, _expected_sha256, _finite_number, _model_path,
    load_model_package,
)

TASKS = ("person_detection", "person_reid", "detection", "recognition")
DOWNLOAD_PACKAGES = ("cheetah_s", "cheetah_l")
DEFAULT_FACE_DET_SIZE = 640


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be an OpenCV BGR uint8 ndarray")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("image must be a nonempty HWC array with three channels")
    return image


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_fingerprint(descriptor):
    """Identify weights and feature semantics, independent of packaging notes."""
    fields = {key: descriptor[key] for key in ("sha256", "task", "input_size", "preprocessing")}
    if descriptor["task"] in ("person_reid", "recognition"):
        fields.update(embedding_dimension=descriptor["embedding_dimension"], normalization="l2")
    if descriptor["task"] in ("detection", "recognition"):
        fields["preprocessing_version"] = descriptor["preprocessing_version"]
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_artifact(descriptor):
    path = Path(descriptor["path"])
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {path}")
    if _digest(path) != descriptor["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _input_size(value, multiple, label):
    if (not isinstance(value, list) or len(value) != 2 or
            any(type(v) is not int or v <= 0 or v % multiple for v in value)):
        raise ValueError(f"{label} input_size must be [height,width], positive multiples of {multiple}")


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} is outside the float32 range") from error
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        effective = np.float32(value)
    if not math.isfinite(value) or not np.isfinite(effective) or positive and effective <= 0:
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''} in float32")
    if label == "std":
        with np.errstate(over="ignore", under="ignore"):
            inverse = np.float32(1.0 / value)
        if not np.isfinite(inverse) or inverse <= 0:
            raise ValueError("std reciprocal must be finite and positive in float32")
    return value


def _check_pixel_range(task, rules):
    """Validate both RGB endpoints once, using the preprocessing operation order."""
    value = np.asarray([[0., 0., 0.], [255., 255., 255.]], np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        if task == "person_detection":
            value *= rules["scale"]
            value -= np.asarray(rules["mean"])
            value /= np.asarray(rules["std"])
        elif task == "person_reid":
            value *= rules["scale"]
            value = (value - np.asarray(rules["mean"], np.float32)) * np.asarray(
                [1.0 / std for std in rules["std"]], np.float32)
        else:
            value -= rules["mean"]
            value *= 1.0 / rules["std"]
    if not np.isfinite(value).all():
        raise ValueError(f"{task}.preprocessing overflows float32 for valid RGB pixels")


def resolve_person_descriptor(task, descriptor):
    """Normalize body rules: (RGB * scale - mean) / std, without changing input metadata."""
    raw = descriptor.get("preprocessing", {})
    if raw == EMBEDDED_PREPROCESSING:
        preprocessing = EMBEDDED_PREPROCESSING
    else:
        if not isinstance(raw, dict) or set(raw) - {"mean", "std", "scale"}:
            raise ValueError(f"{task}.preprocessing accepts embedded or mean, std and scale")
        preprocessing = {"scale": _finite_number(raw.get("scale", 1), f"{task}.scale", positive=True)}
        for key in ("mean", "std"):
            value = raw.get(key, 0 if key == "mean" else 1)
            values = value if isinstance(value, list) else [value] * 3
            if len(values) != 3:
                raise ValueError(f"{task}.{key} must be a number or three RGB values")
            preprocessing[key] = [_finite_number(item, f"{task}.{key}", positive=key == "std") for item in values]
    size = descriptor.get("input_size", [640, 640] if task == "person_detection" else [256, 128])
    _input_size(size, 1, task)
    result = dict(descriptor, task=task, input_size=list(size), preprocessing=preprocessing)
    if task == "person_reid":
        result["embedding_dimension"] = 256
    return result


def _package_path(name, root):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
        raise ValueError("invalid model package name")
    return Path(root).expanduser().resolve() / "models" / name


def load_person_package(name="cheetah_l", root="~/.insightface"):
    """Validate a local manifest, paths and hashes without downloading or opening sessions."""
    folder = _package_path(name, root).resolve()
    # Reuse V2 root, face-task, license-path and optional metadata semantics.
    package = load_model_package(folder)
    raw = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    raw_tasks = raw["tasks"]
    if not set(TASKS) <= set(raw_tasks):
        raise ValueError(f"person model package requires tasks {TASKS}")
    tasks = {}
    for task in TASKS:
        if task.startswith("person_"):
            metadata = raw_tasks[task]
            if not isinstance(metadata, dict):
                raise TypeError(f"tasks.{task} must be an object")
            if "file" not in metadata:
                raise ValueError(f"tasks.{task}.file is required")
            filename, path = _model_path(folder, task, metadata["file"])
            descriptor = {"file": filename, "path": str(path)}
            for key in ("input_size", "preprocessing"):
                if key in metadata:
                    descriptor[key] = metadata[key]
            if "sha256" in metadata:
                descriptor["sha256"] = _expected_sha256(metadata["sha256"], f"tasks.{task}.sha256")
            descriptor = resolve_person_descriptor(task, descriptor)
        else:
            descriptor = package.task(task).as_config()
            descriptor["task"] = task
            if task == "detection":
                # Generic V2 does not consume detection.input_size. PersonAnalysis
                # likewise uses its runtime default, unless face_det_size overrides it.
                descriptor["input_size"] = [DEFAULT_FACE_DET_SIZE, DEFAULT_FACE_DET_SIZE]
        path = Path(descriptor["path"])
        if not path.is_file():
            raise FileNotFoundError(f"model file does not exist: {path}")
        actual_hash = _digest(path)
        if "sha256" in descriptor and actual_hash != descriptor["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        # Even unpinned V2 artifacts need a real identity for reference matching
        # and for detecting changes between metadata loading and Session creation.
        descriptor["sha256"] = actual_hash
        tasks[task] = descriptor
    if len({descriptor["path"] for descriptor in tasks.values()}) != len(tasks):
        raise ValueError("model tasks must reference distinct artifacts")
    return {"manifest_version": package.manifest_version, "model_id": package.model_id,
            "display_name": package.display_name, "license": str(package.license_path.relative_to(folder)),
            "tasks": tasks, "path": str(folder)}


def _download_person_archive(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "InsightFace", "Accept": "application/octet-stream"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _extract_person_archive(archive, destination):
    """Reject ambiguous or escaping archive entries before writing any files."""
    with zipfile.ZipFile(archive) as bundle:
        seen = set()
        for member in bundle.infolist():
            # ZipInfo.filename normalizes Windows separators and truncates NULs.
            filename = member.orig_filename
            path = PurePosixPath(filename)
            if (not path.parts or path.is_absolute() or ".." in path.parts or
                    "\\" in filename or ":" in filename or "\x00" in filename or
                    stat.S_ISLNK(member.external_attr >> 16) or path in seen):
                raise ValueError(f"unsafe model archive entry: {filename!r}")
            seen.add(path)
        bundle.extractall(destination)


def ensure_person_package(name="cheetah_l", root="~/.insightface"):
    """Load local models, or install an absent official Cheetah package atomically."""
    from ..utils.storage import model_zoo_download_url

    folder = _package_path(name, root)

    def local_package():
        try:
            return load_person_package(name, root)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid local model package at {folder}: {exc}. "
                "Repair the local package; it will not be overwritten automatically."
            ) from exc

    if folder.exists() or folder.is_symlink():
        return local_package()
    if name not in DOWNLOAD_PACKAGES:
        raise FileNotFoundError(f"Install the complete local model package at {folder}; automatic download supports cheetah_s and cheetah_l.")
    url = model_zoo_download_url(f"{name}.zip")
    try:
        folder.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{name}-download-", dir=folder.parent) as temporary:
            staging_root = Path(temporary)
            archive = staging_root / f"{name}.zip"
            _download_person_archive(url, archive)
            unpacked = staging_root / "unpacked"
            _extract_person_archive(archive, unpacked)
            package = unpacked if (unpacked / "manifest.json").is_file() else unpacked / name
            staging_package = staging_root / "models" / name
            staging_package.parent.mkdir()
            package.rename(staging_package)
            staged = load_person_package(name, staging_root)
            if staged["model_id"] != name:
                raise ValueError("downloaded manifest model_id does not match package name")
            if not (staging_package / staged["license"]).is_file():
                raise ValueError("downloaded package is missing MODEL.LICENSE")
            if folder.exists() or folder.is_symlink():
                return local_package()
            try:
                staging_package.rename(folder)
            except OSError:
                if folder.exists() or folder.is_symlink():
                    return local_package()
                raise
        return load_person_package(name, root)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download and install {name} from {url}: {exc}. "
            f"Install its complete archive manually at {folder}, with manifest.json directly in that folder."
        ) from exc


def _check_nodes(session, descriptor):
    """Validate runtime metadata against the task contract, then retain it."""
    inputs, outputs = session.get_inputs(), session.get_outputs()
    task = descriptor["task"]
    for nodes in (inputs, outputs):
        if not nodes or len({node.name for node in nodes}) != len(nodes):
            raise ValueError("model nodes must have unique nonempty names")
        for node in nodes:
            embedded_image = node is inputs[0] and descriptor["preprocessing"] == EMBEDDED_PREPROCESSING
            if (not node.name or not node.shape or
                    node.type != "tensor(float)" and not (embedded_image and node.type == "tensor(uint8)")):
                raise ValueError(f"invalid node dtype or shape: {node.name}")
    if task == "person_detection":
        if [n.name for n in inputs] != ["image", "scale_factor"] or [n.name for n in outputs] != ["nms_pre_boxes", "nms_pre_scores"]:
            raise ValueError("person detector node names do not match its task contract")
        in_shapes, out_shapes = [[1, 3, *descriptor["input_size"]], [1, 2]], [[1, None, 4], [1, 1, None]]
    elif task == "person_reid":
        if [n.name for n in inputs] != ["data"] or [n.name for n in outputs] != ["reid_embedding"]:
            raise ValueError("person ReID node names do not match its task contract")
        in_shapes, out_shapes = [[1, 3, *descriptor["input_size"]]], [[1, 256]]
    elif task == "recognition":
        in_shapes, out_shapes = [[1, 3, *descriptor["input_size"]]], [[1, descriptor["embedding_dimension"]]]
    elif task == "detection":
        # Three stride levels, two anchors, and five landmarks are required by these packages.
        if len(outputs) != 9 or len({len(node.shape) for node in outputs}) != 1:
            raise ValueError("face detector requires nine SCRFD outputs including landmarks")
        rank = len(outputs[0].shape)
        if rank not in (2, 3):
            raise ValueError("SCRFD outputs must have rank 2 or batch-1 rank 3")
        in_shapes = [[1, 3, *descriptor["input_size"]]]
        out_shapes = [([1] if rank == 3 else []) + [None, width] for width in (1, 4, 10) for _ in range(3)]
    else:
        raise ValueError(f"unsupported task: {task}")
    for nodes, expected in ((inputs, in_shapes), (outputs, out_shapes)):
        if len(nodes) != len(expected):
            raise ValueError(f"invalid {task} input/output count")
        for node, shape in zip(nodes, expected):
            if len(node.shape) != len(shape) or any(
                    isinstance(actual, int) and (actual <= 0 or required is not None and actual != required)
                    for actual, required in zip(node.shape, shape)):
                raise ValueError(f"invalid {task} node shape: {node.name}")
            if task in ("person_reid", "recognition") and node.shape[1:] != shape[1:]:
                raise ValueError(f"feature dimensions must be fixed for {node.name}")
    for kind, nodes in (("inputs", inputs), ("outputs", outputs)):
        descriptor[kind] = [{"name": n.name, "shape": list(n.shape)} for n in nodes]
    descriptor["input_dtype"] = "uint8" if inputs[0].type == "tensor(uint8)" else "float32"
    return in_shapes


def _check_warmup(descriptor, values):
    """Check actual outputs too: dynamic dimensions cannot establish the contract."""
    task = descriptor["task"]
    if len(values) != len(descriptor["outputs"]) or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in values):
        raise ValueError("invalid model warmup output dtype or values")
    if task == "person_detection":
        valid = (values[0].ndim == 3 and values[0].shape[:1] == (1,) and values[0].shape[-1] == 4 and
                 values[1].shape == (1, 1, values[0].shape[1]))
    elif task in ("person_reid", "recognition"):
        valid = values[0].shape == (1, descriptor["embedding_dimension"])
    else:
        height, width = descriptor["input_size"]
        shapes = []
        for dimension in (1, 4, 10):
            for stride in (8, 16, 32):
                shapes.append((height // stride * (width // stride) * 2, dimension))
        batched = len(descriptor["outputs"][0]["shape"]) == 3
        valid = all(value.shape == ((1, *expected) if batched else expected)
                    for value, expected in zip(values, shapes))
    if not valid:
        raise ValueError(f"invalid {task} warmup output shape")


def _validate_runtime_descriptor(descriptor):
    """Check adapter execution limits without narrowing the V2 manifest schema."""
    task = descriptor["task"]
    multiple = 64 if task == "person_detection" else 32 if task == "detection" else 1
    _input_size(descriptor["input_size"], multiple, task)
    if task == "recognition":
        height, width = descriptor["input_size"]
        if height != width or height % 112 and height % 128:
            raise ValueError("face recognition alignment requires a square input size divisible by 112 or 128")
    preprocessing = descriptor["preprocessing"]
    if task.startswith("person_") and preprocessing != EMBEDDED_PREPROCESSING:
        for key, value in preprocessing.items():
            for item in value if isinstance(value, list) else [value]:
                _number(item, key, key in ("std", "scale"))
        _check_pixel_range(task, preprocessing)


def create_session(descriptor, providers, threads=4, static_scrfd=False):
    """Read actual model nodes and validate one warmup at the resolved input size."""
    import onnx
    import onnxruntime as ort
    from .onnxruntime_utils import preload_cuda_libraries

    verify_artifact(descriptor)
    _validate_runtime_descriptor(descriptor)
    if type(threads) is not int or threads <= 0:
        raise ValueError("threads must be a positive integer")
    if not providers or isinstance(providers, str):
        raise ValueError("providers must be a nonempty sequence")
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    if any(p not in {"CPUExecutionProvider", "CUDAExecutionProvider"} for p in names):
        raise ValueError("PersonAnalysis supports CPUExecutionProvider and CUDAExecutionProvider")
    available = ort.get_available_providers()
    missing = set(names) - set(available)
    if missing:
        raise RuntimeError(f"requested ONNX providers unavailable: {sorted(missing)}; available: {available}")
    preload_cuda_libraries(providers)
    configured = [(p, {"use_tf32": "0"}) if p == "CUDAExecutionProvider" else p for p in providers]
    graph = onnx.load(descriptor["path"], load_external_data=False)
    if descriptor["task"] in ("person_detection", "detection") and any(n.op_type == "NonMaxSuppression" for n in graph.graph.node):
        raise ValueError("detector task requires candidates without in-graph NonMaxSuppression")
    model_source = descriptor["path"]
    if static_scrfd:
        from .scrfd import _static_scrfd_model
        if len(graph.graph.input) != 1:
            raise ValueError("SCRFD requires one image input")
        model_source = _static_scrfd_model(model_source, tuple(reversed(descriptor["input_size"])), graph.graph.input[0].name)
    with tempfile.TemporaryDirectory(prefix="insightface-person-ort-") as profile_dir:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(profile_dir) / "warmup")
        session = ort.InferenceSession(model_source, sess_options=options, providers=configured)
        session.disable_fallback()
        active = session.get_providers()
        if any(p not in active for p in names):
            raise RuntimeError(f"requested providers {names} were not activated; actual providers: {active}")
        input_shapes = _check_nodes(session, descriptor)
        feed = {node["name"]: np.ones(shape, descriptor["input_dtype"] if index == 0 else np.float32)
                for index, (node, shape) in enumerate(zip(descriptor["inputs"], input_shapes))}
        _check_warmup(descriptor, session.run([n["name"] for n in descriptor["outputs"]], feed))
        profile = json.loads(Path(session.end_profiling()).read_text())
        placement = Counter(e.get("args", {}).get("provider") for e in profile if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
    diagnostics = {
        "requested_providers": names, "active_providers": active,
        "warmup_node_provider_counts": dict(placement), "cpu_threads": threads,
        "cpu_node_fallback_observed": "CUDAExecutionProvider" in names and placement.get("CPUExecutionProvider", 0) > 0,
        "placement_scope": "One prepare-time warmup at the resolved input size; counts are executed node events, not a guarantee for every dynamic shape.",
        "sha256": descriptor["sha256"], "model_id": model_fingerprint(descriptor),
    }
    return session, diagnostics


def prepare_models(name="cheetah_l", root="~/.insightface", providers=None, threads=4,
                   body_det_size=0, face_det_size=0):
    """Load the four tasks only when PersonAnalysis.prepare calls us."""
    from .person_detection import PersonDetection
    from .person_reid import PersonReID
    from .scrfd import SCRFD
    from .arcface_onnx import ArcFaceONNX
    from .model_zoo import _configure_image_preprocessing

    def configure_face(model, session, task):
        preprocessing = tasks[task]["preprocessing"]
        mean, std = ((0., 1.) if preprocessing == EMBEDDED_PREPROCESSING else
                     (preprocessing["mean"], preprocessing["std"]))
        return _configure_image_preprocessing(model, session, task, preprocessing, mean, std)

    package = ensure_person_package(name, root)
    tasks = package["tasks"]
    if body_det_size:
        tasks["person_detection"]["input_size"] = [body_det_size, body_det_size]
    if face_det_size:
        tasks["detection"]["input_size"] = [face_det_size, face_det_size]
    providers = ["CPUExecutionProvider"] if providers is None else providers
    detector = PersonDetection(tasks["person_detection"], providers, threads)
    reid = PersonReID(tasks["person_reid"], providers, threads)
    # Current SCRFD exports carry 640-sized output metadata; this is not an input-size fallback.
    face_det_session, face_det_diagnostics = create_session(tasks["detection"], providers, threads,
        static_scrfd=bool(face_det_size))
    face_rec_session, face_rec_diagnostics = create_session(tasks["recognition"], providers, threads)
    face_detector = SCRFD(tasks["detection"]["path"], session=face_det_session, static_shape_sessions=False)
    configure_face(face_detector, face_det_session, "detection")
    face_detector.prepare(ctx_id=0, input_size=tuple(reversed(tasks["detection"]["input_size"])), det_thresh=0.5)
    face_recognizer = ArcFaceONNX(tasks["recognition"]["path"], session=face_rec_session)
    configure_face(face_recognizer, face_rec_session, "recognition")
    return {"detector": detector, "reid": reid, "face_detector": face_detector,
            "face_recognizer": face_recognizer,
            "face_model_id": model_fingerprint(tasks["recognition"]), "reid_model_id": reid.model_id,
            "diagnostics": {"package": name, "person_detection": detector.diagnostics,
                "person_reid": reid.diagnostics, "face_detection": face_det_diagnostics,
                "face_recognition": face_rec_diagnostics}}
