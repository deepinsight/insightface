# -*- coding: utf-8 -*-
"""ONNX-based InsightFace face swapping model."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import onnxruntime
import cv2
import onnx
from onnx import numpy_helper

from ..utils import face_align

__all__ = ['INSwapper']


class INSwapper:
    """ONNX-based face swapping model.
    
    This class provides face swapping functionality using InsightFace models.
    
    Attributes:
        model_file: Path to the ONNX model file.
        session: ONNX Runtime inference session.
        emap: Embedding mapping matrix.
        input_mean: Mean value for input normalization.
        input_std: Std value for input normalization.
        input_size: Expected input size (width, height).
        input_shape: Full input shape.
    """
    
    def __init__(
        self,
        model_file: Optional[str] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> None:
        """Initialize the INSwapper model.
        
        Args:
            model_file: Path to the ONNX model file.
            session: Pre-existing ONNX Runtime session (optional).
        """
        self.model_file = model_file
        self.session = session
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0
        
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        
        inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in inputs]
        
        outputs = self.session.get_outputs()
        self.output_names = [out.name for out in outputs]
        assert len(self.output_names) == 1
        
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print('inswapper-shape:', self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])

    def forward(
        self,
        img: np.ndarray,
        latent: np.ndarray,
    ) -> np.ndarray:
        """Run forward pass on image with latent vector.
        
        Args:
            img: Input image tensor (normalized).
            latent: Latent embedding vector.
        
        Returns:
            Model prediction output.
        """
        img = (img - self.input_mean) / self.input_std
        pred = self.session.run(
            self.output_names,
            {self.input_names[0]: img, self.input_names[1]: latent}
        )[0]
        return pred

    def get(
        self,
        img: np.ndarray,
        target_face: Any,
        source_face: Any,
        paste_back: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Perform face swap from source face to target face.
        
        Args:
            img: Input image as numpy array.
                - Format: BGR (OpenCV default)
                - Dtype: uint8
                - Range: 0-255
                - Shape: (H, W, 3)
            target_face: Face object for target face (where to swap).
            source_face: Face object for source face (what to swap).
            paste_back: Whether to paste result back to original image.
        
        Returns:
            If paste_back is True: Swapped image with face blended.
            If paste_back is False: Tuple of (swapped_face_crop, transform_matrix).
        """
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(
            self.output_names,
            {self.input_names[0]: blob, self.input_names[1]: latent}
        )[0]
        
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        
        if not paste_back:
            return bgr_fake, M
        
        target_img = img
        fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0
        
        IM = cv2.invertAffineTransform(M)
        img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
        bgr_fake = cv2.warpAffine(
            bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0
        )
        img_white = cv2.warpAffine(
            img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0
        )
        fake_diff = cv2.warpAffine(
            fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0
        )
        
        img_white[img_white > 20] = 255
        fthresh = 10
        fake_diff[fake_diff < fthresh] = 0
        fake_diff[fake_diff >= fthresh] = 255
        img_mask = img_white
        
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        
        kernel = np.ones((2, 2), np.uint8)
        fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
        
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        
        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
        
        img_mask /= 255
        fake_diff /= 255
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
        fake_merged = fake_merged.astype(np.uint8)
        
        return fake_merged
