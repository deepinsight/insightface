# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      :


from __future__ import division

import glob
import logging
import os.path as osp
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime

from ..addons import ADDON_CATALOG, Liveness, ensure_addon
from ..addons.liveness import DEFAULT_THRESHOLD, validate_threshold
from ..model_zoo import model_zoo
from ..model_zoo.onnxruntime_utils import get_default_providers
from ..model_zoo.package_manifest import (
    ModelPackageDescriptor,
    has_model_package_manifest,
    load_model_package,
)
from ..utils import DEFAULT_MP_NAME, ensure_available
from .common import Face

__all__ = ["FaceAnalysis"]

logger = logging.getLogger(__name__)

DEFAULT_DET_SIZES = [(128, 128), (640, 640)]


def _provider_name(provider):
    if isinstance(provider, (list, tuple)) and provider:
        return str(provider[0])
    return str(provider)


def _default_coreml_detector_input_size():
    return max(
        DEFAULT_DET_SIZES,
        key=lambda value: int(value[0]) * int(value[1]),
    )


def _is_auto_det_size(det_size):
    if det_size is None:
        return True
    if isinstance(det_size, np.ndarray):
        det_size = det_size.tolist()
    if isinstance(det_size, (list, tuple)) and len(det_size) == 2:
        first, second = det_size
        if not isinstance(first, (list, tuple, np.ndarray)) and not isinstance(
            second, (list, tuple, np.ndarray)
        ):
            try:
                return int(first) == 0 and int(second) == 0
            except (TypeError, ValueError):
                return False
    return False


class FaceAnalysis:
    def __init__(
        self,
        name=DEFAULT_MP_NAME,
        root="~/.insightface",
        allowed_modules=None,
        *,
        addons=None,
        liveness_mode="normal",
        liveness_threshold=DEFAULT_THRESHOLD,
        **kwargs,
    ):
        if addons is None:
            addons = ()
        if isinstance(addons, str) or not isinstance(addons, Sequence):
            raise TypeError("addons must be a sequence of addon names, e.g. ['liveness']")
        for addon in addons:
            if not isinstance(addon, str) or addon not in ADDON_CATALOG:
                raise ValueError(f"Unknown addon {addon!r}; available: {list(ADDON_CATALOG)}")
        if len(set(addons)) != len(addons):
            raise ValueError("addons must not contain duplicate names")
        if liveness_mode not in ("normal", "observe"):
            raise ValueError("liveness_mode must be normal or observe")
        self.liveness_mode = liveness_mode
        self.liveness_threshold = validate_threshold(liveness_threshold)
        self.addons = {}
        onnxruntime.set_default_logger_severity(3)
        if kwargs.get("providers") is None:
            kwargs["providers"] = get_default_providers()
        providers = kwargs.get("providers") or ()
        if providers and _provider_name(providers[0]) == "CoreMLExecutionProvider":
            # CoreML cannot safely construct the dynamic SCRFD Session with
            # the default ALL compute-unit policy on some ORT/macOS versions.
            # Pass an internal fixed main resolution to both manifest and
            # legacy model loading.
            if kwargs.get("static_shape_sessions", True) is not False:
                kwargs.setdefault(
                    "_coreml_detector_input_size",
                    _default_coreml_detector_input_size(),
                )
        self.models = {}
        self.det_model = None
        self.model_package = None

        if isinstance(name, ModelPackageDescriptor):
            self.model_package = name
            self.model_dir = str(name.path)
        else:
            direct = Path(str(name)).expanduser()
            if direct.is_dir():
                self.model_dir = str(direct.resolve())
            else:
                self.model_dir = ensure_available("models", name, root=root)

        if self.model_package is not None or has_model_package_manifest(self.model_dir):
            self._load_manifest_package(
                allowed_modules,
                kwargs,
            )
            assert "detection" in self.models
            self.det_model = self.models["detection"]
            self._load_addons(addons, root, kwargs)
            return

        onnx_files = glob.glob(osp.join(self.model_dir, "*.onnx"))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                logger.debug("model not recognized: %s", onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                logger.debug("model ignore: %s %s", onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (
                allowed_modules is None or model.taskname in allowed_modules
            ):
                logger.debug(
                    "find model: %s %s %s %s %s",
                    onnx_file,
                    model.taskname,
                    model.input_shape,
                    model.input_mean,
                    model.input_std,
                )
                self.models[model.taskname] = model
            else:
                logger.debug(
                    "duplicated model task type, ignore: %s %s",
                    onnx_file,
                    model.taskname,
                )
                del model
        assert "detection" in self.models
        self.det_model = self.models["detection"]
        self._load_addons(addons, root, kwargs)

    def _load_addons(self, addons, root, kwargs):
        if "liveness" in addons:
            self.addons["liveness"] = Liveness(
                ensure_addon("liveness", root=root),
                threshold=self.liveness_threshold,
                providers=kwargs.get("providers"),
                provider_options=kwargs.get("provider_options"),
                sess_options=kwargs.get("sess_options"),
            )

    def _load_manifest_package(
        self,
        allowed_modules,
        kwargs,
    ):
        # Parse once before any Session construction. Invalid manifests must
        # never fall back to filename/shape heuristics.
        if self.model_package is None:
            self.model_package = load_model_package(self.model_dir)
        requested = None if allowed_modules is None else set(allowed_modules)

        model_kwargs = dict(kwargs)

        # The descriptor is the source of truth for available tasks.
        # ``allowed_modules`` only filters that declared set.
        for task in self.model_package.tasks:
            if requested is not None and task not in requested:
                continue
            descriptor = self.model_package.task(task)
            model = model_zoo.get_model(
                self.model_dir,
                model_task=task,
                model_descriptor=descriptor,
                **model_kwargs,
            )
            if model.taskname != task:
                raise RuntimeError(
                    f"manifest {task} model reported task {model.taskname}, "
                    f"expected {task}"
                )
            if task in self.models:
                raise RuntimeError(f"duplicated manifest model task: {task}")
            logger.debug(
                "find manifest model: %s %s %s %s %s",
                descriptor.path,
                model.taskname,
                getattr(model, "input_shape", None),
                getattr(model, "input_mean", None),
                getattr(model, "input_std", None),
            )
            self.models[task] = model

    def prepare(self, ctx_id=0, det_thresh=0.5, det_size=None):
        self.det_thresh = det_thresh
        if _is_auto_det_size(det_size):
            det_size = list(DEFAULT_DET_SIZES)
        logger.debug("set det-size: %s", det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == "detection":
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)
        for model in self.addons.values():
            model.prepare(ctx_id)

    def get(self, img, max_num=0, det_metric="default"):
        """Detect faces and run enabled addons before recognition.

        Liveness returns status, is_live and live_score on each Face, with an
        English reason when the aligned crop exceeds the image bounds.
        In normal mode, rejected inputs and fake faces retain their detection
        result but do not receive an embedding. Other selected tasks still run.
        Without the addon, the liveness field is absent and recognition runs
        as before, regardless of the selected liveness mode.
        Invalid landmarks, alignment and inference failures raise exceptions
        in both modes instead of returning a liveness classification.
        """
        liveness = self.addons.get("liveness")
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric=det_metric)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            if liveness is not None:
                liveness.get(img, face)
            for taskname, model in self.models.items():
                if taskname == "detection":
                    continue
                if (
                    taskname == "recognition"
                    and liveness is not None
                    and self.liveness_mode == "normal"
                    and face.liveness.is_live is not True
                ):
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2

        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                # print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
            if face.gender is not None and face.age is not None:
                cv2.putText(
                    dimg,
                    "%s,%d" % (face.sex, face.age),
                    (box[0] - 1, box[1] - 4),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )

            # for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg
