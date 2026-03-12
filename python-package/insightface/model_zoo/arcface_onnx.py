# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import cv2
import onnx
import onnxruntime

from ..utils import face_align

__all__ = ['ArcFaceONNX']


class ArcFaceONNX:
    """ONNX-based ArcFace face recognition model.
    
    This class provides face embedding extraction using ArcFace models.
    
    Attributes:
        model_file: Path to the ONNX model file.
        session: ONNX Runtime inference session.
        taskname: Task identifier ('recognition').
        input_size: Expected input size (width, height).
        input_mean: Mean value for input normalization.
        input_std: Std value for input normalization.
    """
    
    def __init__(
        self,
        model_file: Optional[str] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> None:
        """Initialize the ArcFace model.
        
        Args:
            model_file: Path to the ONNX model file.
            session: Pre-existing ONNX Runtime session (optional).
        """
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'
        
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        
        if find_sub and find_mul:
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        
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
        self.output_shape = outputs[0].shape

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
    ) -> np.ndarray:
        """Extract face embedding from image.
        
        Args:
            img: Input image as numpy array.
                - Format: BGR (OpenCV default)
                - Dtype: uint8
                - Range: 0-255
                - Shape: (H, W, 3)
            face: Face object containing keypoints for alignment.
        
        Returns:
            Flattened embedding vector.
        """
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def compute_sim(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two feature vectors.
        
        Args:
            feat1: First feature vector.
            feat2: Second feature vector.
        
        Returns:
            Cosine similarity score in range [-1, 1].
        """
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return float(sim)

    def get_feat(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """Extract features from preprocessed images.
        
        Args:
            imgs: Single image or list of images.
                - Format: BGR
                - Dtype: uint8
                - Range: 0-255
        
        Returns:
            Feature vectors with shape (N, embedding_dim).
        """
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        
        blob = cv2.dnn.blobFromImages(
            imgs,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data: np.ndarray) -> np.ndarray:
        """Run forward pass on pre-normalized batch data.
        
        Args:
            batch_data: Pre-normalized input tensor with shape (N, C, H, W).
        
        Returns:
            Feature vectors.
        """
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out
