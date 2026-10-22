"""Face swap model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ...model_zoo.onnxruntime_utils import get_default_providers


class GFPGANRestorer:
    """Small ONNXRuntime wrapper for GFPGAN 512x512 face restoration."""

    def __init__(self, model_path: str = "", providers: Optional[list[str]] = None):
        self.model_path = model_path
        requested_providers = list(providers) if providers is not None else []
        self.providers = requested_providers or get_default_providers()
        self.session = None
        self.input_name = ""
        self.output_name = ""
        self.last_error = ""

    def load(self) -> bool:
        path = Path(self.model_path).expanduser()
        if not path.exists():
            self.last_error = "GFPGAN model not found. Download GFPGANv1.4.onnx in Models > Downloads first."
            return False
        try:
            import onnxruntime

            onnxruntime.set_default_logger_severity(3)
            self.session = onnxruntime.InferenceSession(str(path), providers=self.providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            return True
        except Exception as exc:
            self.last_error = f"GFPGAN model load failed: {exc}"
            return False

    def restore(self, image: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("GFPGAN model is not loaded.")
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError(f"OpenCV is required for GFPGAN post-processing: {exc}") from exc

        original_height, original_width = image.shape[:2]
        resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None, :, :, :]
        outputs = self.session.run([self.output_name], {self.input_name: tensor.astype(np.float32)})
        restored = np.asarray(outputs[0])
        if restored.ndim == 4:
            restored = restored[0]
        if restored.shape[0] == 3:
            restored = restored.transpose(1, 2, 0)
        restored = restored.astype(np.float32)
        if float(restored.min()) < 0.0:
            restored = (restored + 1.0) * 0.5
        restored = np.clip(restored, 0.0, 1.0)
        restored = (restored * 255.0).round().astype(np.uint8)
        bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        if (original_width, original_height) != (512, 512):
            bgr = cv2.resize(bgr, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        return bgr


class FaceSwapEngine:
    def __init__(
        self,
        model_path: str = "",
        providers: Optional[list[str]] = None,
        gfpgan_model_path: str = "",
        enable_gfpgan: bool = False,
    ):
        self.model_path = model_path
        requested_providers = list(providers) if providers is not None else []
        self.providers = requested_providers or get_default_providers()
        self.model = None
        self.enable_gfpgan = enable_gfpgan
        self.restorer = GFPGANRestorer(gfpgan_model_path, self.providers) if enable_gfpgan else None
        self.last_error = ""

    def is_available(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        path = Path(self.model_path).expanduser()
        if not path.exists():
            self.last_error = (
                "Face swap model not found. Please configure a valid swap model in Models."
            )
            return False
        try:
            import onnxruntime
            from insightface.model_zoo import get_model

            onnxruntime.set_default_logger_severity(3)
            self.model = get_model(str(path), providers=self.providers)
            if self.model is None:
                self.last_error = "Configured face swap model could not be recognized."
                return False
            if self.restorer is not None and not self.restorer.load():
                self.last_error = self.restorer.last_error
                return False
            return True
        except Exception as exc:
            self.last_error = f"Face swap model load failed: {exc}"
            return False

    def swap(self, image: np.ndarray, target_face, source_face) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Face swap model not found. Please configure a valid swap model in Models.")
        output = self.model.get(image, target_face, source_face, paste_back=True)
        if self.restorer is not None:
            output = self.restorer.restore(output)
        return output
