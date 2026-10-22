"""80x80 RGB liveness adapter with a fixed aligned-mask input gate.

The model is distributed separately from the Python package. Inputs to this
adapter follow FaceAnalysis's BGR image convention; RGB/255 conversion happens
only after the dedicated five-point alignment. The ONNX output is already a
live probability and must not be passed through another softmax.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

from ..model_zoo.model_zoo import PickableInferenceSession

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.8
MAX_OOB_RATIO = 0.30
INPUT_SIZE = 80
OUT_OF_BOUNDS_REASON = (
    "Insufficient image area around the face for liveness detection. "
    "Move the face toward the center, step back from the camera, "
    "or use a less tightly cropped image."
)
DESTINATION_LANDMARKS = np.array(
    [[32.03, 38.06], [47.89, 37.98], [40.01, 47.08], [33.50, 56.36], [46.63, 56.29]],
    dtype=np.float32,
)
DESTINATION_LANDMARKS.setflags(write=False)


def validate_threshold(value):
    if isinstance(value, (bool, str, bytes)):
        raise TypeError("liveness_threshold must be a number between 0 and 1")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "liveness_threshold must be a number between 0 and 1"
        ) from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("liveness_threshold must be a number between 0 and 1")
    return threshold


def _input_rejected(oob_ratio):
    LOGGER.debug(
        "Liveness input rejected: aligned_crop_out_of_bounds oob_ratio=%s", oob_ratio
    )
    return {
        "status": "input_rejected",
        "is_live": None,
        "live_score": None,
        "reason": OUT_OF_BOUNDS_REASON,
    }


class Liveness:
    """Evaluate one detected face without running detection or recognition.

    A supplied session is reused, allowing applications to own its lifecycle.
    Excessive crop missing area returns input_rejected with an English reason.
    Invalid images or landmarks, failed alignment, incompatible models and
    inference failures raise exceptions.
    """

    taskname = "liveness"

    def __init__(
        self,
        model_file,
        *,
        session=None,
        threshold=DEFAULT_THRESHOLD,
        providers=None,
        provider_options=None,
        sess_options=None,
    ):
        self.threshold = validate_threshold(threshold)
        self.model_file = str(model_file)
        self.session = (
            session
            if session is not None
            else PickableInferenceSession(
                self.model_file,
                providers=providers,
                provider_options=provider_options,
                sess_options=sess_options,
            )
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("Liveness model must have one input and one output")
        input_meta, output_meta = inputs[0], outputs[0]
        self.input_shape = list(input_meta.shape)
        self.output_shape = list(output_meta.shape)
        if (
            input_meta.type != "tensor(float)"
            or len(self.input_shape) != 4
            or self.input_shape[1:] != [3, INPUT_SIZE, INPUT_SIZE]
            or (isinstance(self.input_shape[0], int) and self.input_shape[0] != 1)
            or output_meta.type != "tensor(float)"
            or len(self.output_shape) != 1
            or (isinstance(self.output_shape[0], int) and self.output_shape[0] != 1)
        ):
            raise RuntimeError(
                "Liveness model requires float32 [N, 3, 80, 80] input and [N] live probability output"
            )
        self.input_name = input_meta.name
        self.output_name = output_meta.name

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])

    def predict(self, image, landmarks):
        """Return status, is_live and live_score, plus reason for rejected crops.

        The reason is English display text; use status/is_live for decisions.
        """

        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
            or image.dtype != np.uint8
        ):
            raise ValueError("Liveness image must be a non-empty uint8 HxWx3 BGR array")
        try:
            source = np.asarray(landmarks, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Liveness landmarks must contain five finite 2D points"
            ) from error
        if source.shape != (5, 2) or not np.all(np.isfinite(source)):
            raise ValueError("Liveness landmarks must contain five finite 2D points")
        try:
            if hasattr(SimilarityTransform, "from_estimate"):
                transform = SimilarityTransform.from_estimate(source, DESTINATION_LANDMARKS)
                if not transform:
                    raise RuntimeError("Liveness face alignment failed")
            else:
                transform = SimilarityTransform()
                if not transform.estimate(source, DESTINATION_LANDMARKS):
                    raise RuntimeError("Liveness face alignment failed")
            matrix = np.asarray(transform.params[:2, :], dtype=np.float64)
        except (ValueError, np.linalg.LinAlgError) as error:
            raise RuntimeError("Liveness face alignment failed") from error
        if not np.all(np.isfinite(matrix)) or np.linalg.det(matrix[:, :2]) <= 0:
            raise RuntimeError("Liveness face alignment failed")

        valid_output = cv2.warpAffine(
            np.ones(image.shape[:2], dtype=np.float32),
            matrix,
            (INPUT_SIZE, INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        oob_ratio = float(np.clip(1.0 - np.mean(valid_output), 0.0, 1.0))
        if oob_ratio > MAX_OOB_RATIO:
            return _input_rejected(oob_ratio)

        crop = cv2.warpAffine(
            image,
            matrix,
            (INPUT_SIZE, INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        tensor = np.ascontiguousarray(
            crop[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        )
        tensor /= 255.0
        scores = np.asarray(
            self.session.run([self.output_name], {self.input_name: tensor})[0]
        )
        if scores.shape != (1,):
            raise RuntimeError(
                f"Liveness model returned invalid score shape: {scores.shape}"
            )
        score = float(scores[0])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"Liveness model returned invalid probability: {score}")
        return {"status": "ok", "is_live": score >= self.threshold, "live_score": score}

    def get(self, image, face):
        face.liveness = self.predict(image, face.kps)
        return face.liveness
