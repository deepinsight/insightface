# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-06-19
# @Function      : 

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import cv2
import onnx
import onnxruntime

from ..utils import face_align

__all__ = ['Attribute']


class Attribute:
    """ONNX-based face attribute detection model.
    
    This class provides gender and age prediction for detected faces.
    
    Attributes:
        model_file: Path to the ONNX model file.
        session: ONNX Runtime inference session.
        taskname: Task identifier ('genderage' or 'attribute_N').
        input_size: Expected input size (width, height).
        input_mean: Mean value for input normalization.
        input_std: Std value for input normalization.
    """
    
    def __init__(
        self,
        model_file: Optional[str] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> None:
        """Initialize the Attribute model.
        
        Args:
            model_file: Path to the ONNX model file.
            session: Pre-existing ONNX Runtime session (optional).
        """
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
            if nid < 3 and node.name == 'bn_data':
                find_sub = True
                find_mul = True
        
        if find_sub and find_mul:
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 128.0
        
        self.input_mean = input_mean
        self.input_std = input_std
        
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        
        outputs = self.session.get_outputs()
        output_names = [out.name for out in outputs]
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        
        output_shape = outputs[0].shape
        if output_shape[1] == 3:
            self.taskname = 'genderage'
        else:
            self.taskname = 'attribute_%d' % output_shape[1]

    def prepare(self, ctx_id: int, **kwargs: Any) -> None:
        """Prepare the model for inference.
        
        Args:
            ctx_id: Context ID for GPU device. Use -1 for CPU.
            **kwargs: Additional arguments (unused).
        """
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(
        self,
        img: np.ndarray,
        face: Any,
    ) -> Union[Tuple[int, int], np.ndarray]:
        """Predict gender and age for a face.
        
        Args:
            img: Input image as numpy array.
                - Format: BGR (OpenCV default)
                - Dtype: uint8
                - Range: 0-255
                - Shape: (H, W, 3)
            face: Face object containing bounding box.
        
        Returns:
            For genderage task: Tuple of (gender, age) where
                - gender: 0 for female, 1 for male
                - age: estimated age in years
            For other tasks: Raw prediction array.
        """
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h) * 1.5)
        
        aimg, M = face_align.transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0]
        
        if self.taskname == 'genderage':
            assert len(pred) == 3
            gender = int(np.argmax(pred[:2]))
            age = int(np.round(pred[2] * 100))
            face['gender'] = gender
            face['age'] = age
            return gender, age
        else:
            return pred
