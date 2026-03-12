# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

from __future__ import annotations

import glob
import os.path as osp
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import onnx
import onnxruntime

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available
from .common import Face
from .visualizer import draw_faces

__all__ = ['FaceAnalysis']


def _safe_slice_bbox(bboxes: np.ndarray, index: int) -> Optional[np.ndarray]:
    """Safely extract bounding box from detection result.
    
    Args:
        bboxes: Detection result array with shape (N, M) where M >= 5.
        index: Index of the detection to extract.
    
    Returns:
        Bounding box array [x1, y1, x2, y2] or None if extraction fails.
    """
    if bboxes is None or bboxes.ndim != 2:
        return None
    if index < 0 or index >= bboxes.shape[0]:
        return None
    if bboxes.shape[1] < 4:
        return None
    return bboxes[index, :4].copy()


def _safe_slice_score(bboxes: np.ndarray, index: int) -> float:
    """Safely extract detection score from detection result.
    
    Args:
        bboxes: Detection result array with shape (N, M) where M >= 5.
        index: Index of the detection to extract.
    
    Returns:
        Detection score or 0.0 if extraction fails.
    """
    if bboxes is None or bboxes.ndim != 2:
        return 0.0
    if index < 0 or index >= bboxes.shape[0]:
        return 0.0
    if bboxes.shape[1] < 5:
        return 0.0
    return float(bboxes[index, 4])


def _safe_slice_kps(kpss: Optional[np.ndarray], index: int) -> Optional[np.ndarray]:
    """Safely extract keypoints from detection result.
    
    Args:
        kpss: Keypoints array with shape (N, K, 2) or None.
        index: Index of the detection to extract.
    
    Returns:
        Keypoints array with shape (K, 2) or None if extraction fails.
    """
    if kpss is None:
        return None
    if kpss.ndim != 3:
        return None
    if index < 0 or index >= kpss.shape[0]:
        return None
    return kpss[index].copy()


class FaceAnalysis:
    """Face Analysis Pipeline.
    
    This class provides a unified interface for face detection, alignment,
    and feature extraction using multiple ONNX models.
    
    Attributes:
        models: Dictionary mapping task names to model instances.
        det_model: The detection model instance.
        model_dir: Directory containing the ONNX model files.
    
    Example:
        >>> app = FaceAnalysis(name='buffalo_l')
        >>> app.prepare(ctx_id=0, det_size=(640, 640))
        >>> faces = app.get(img)
        >>> for face in faces:
        ...     print(face.bbox, face.det_score)
    """
    
    _MODEL_SIGNATURES = {
        'detection': lambda inputs, outputs: len(outputs) >= 5,
        'landmark_2d_106': lambda inputs, outputs: (
            len(inputs) > 0 and 
            len(inputs[0].shape) >= 4 and
            inputs[0].shape[2] == 192 and 
            inputs[0].shape[3] == 192
        ),
        'genderage': lambda inputs, outputs: (
            len(inputs) > 0 and 
            len(inputs[0].shape) >= 4 and
            inputs[0].shape[2] == 96 and 
            inputs[0].shape[3] == 96
        ),
        'inswapper': lambda inputs, outputs: (
            len(inputs) == 2 and 
            len(inputs[0].shape) >= 4 and
            inputs[0].shape[2] == 128 and 
            inputs[0].shape[3] == 128
        ),
        'recognition': lambda inputs, outputs: (
            len(inputs) > 0 and 
            len(inputs[0].shape) >= 4 and
            inputs[0].shape[2] == inputs[0].shape[3] and
            inputs[0].shape[2] >= 112 and 
            inputs[0].shape[2] % 16 == 0
        ),
    }
    
    def __init__(
        self,
        name: str = DEFAULT_MP_NAME,
        root: str = '~/.insightface',
        allowed_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FaceAnalysis.
        
        Args:
            name: Name of the model pack to load. Defaults to 'buffalo_l'.
            root: Root directory for model storage. Defaults to '~/.insightface'.
            allowed_modules: List of module names to load. If None, loads all
                available modules. Common values: ['detection', 'recognition', 
                'landmark_2d_106', 'genderage'].
            **kwargs: Additional arguments passed to model initialization.
        """
        onnxruntime.set_default_logger_severity(3)
        self.models: Dict[str, Any] = {}
        self._model_paths: Dict[str, str] = {}
        self._loaded_models: Set[str] = set()
        self._allowed_modules: Optional[Set[str]] = None
        self._init_kwargs = kwargs
        
        if allowed_modules is not None:
            self._allowed_modules = set(allowed_modules)
        
        self.model_dir = ensure_available('models', name, root=root)
        self._scan_model_files()
        
        assert 'detection' in self._model_paths, \
            "Detection model is required but not found"
        self.det_model = None
    
    def _scan_model_files(self) -> None:
        """Scan and register model files without loading them.
        
        Uses ONNX metadata to identify model types, avoiding full model loading.
        """
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        
        for onnx_file in onnx_files:
            task_name = self._identify_model_task_fast(onnx_file)
            if task_name is None:
                print('model not recognized:', onnx_file)
                continue
            
            if self._allowed_modules is not None and task_name not in self._allowed_modules:
                print('model ignore:', onnx_file, task_name)
                continue
            
            if task_name in self._model_paths:
                print('duplicated model task type, ignore:', onnx_file, task_name)
                continue
            
            print('find model:', onnx_file, task_name)
            self._model_paths[task_name] = onnx_file
    
    def _identify_model_task_fast(self, onnx_file: str) -> Optional[str]:
        """Identify model type by reading ONNX metadata only (no weight loading).
        
        This method reads only the model graph structure without loading weights,
        making it much faster than creating an InferenceSession.
        
        Args:
            onnx_file: Path to the ONNX model file.
        
        Returns:
            Task name string or None if unrecognized.
        """
        try:
            model = onnx.load(onnx_file, load_external_data=False)
            graph = model.graph
            
            inputs = list(graph.input)
            outputs = list(graph.output)
            
            input_shapes = []
            for inp in inputs:
                shape = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(-1)
                input_shapes.append(shape)
            
            output_count = len(outputs)
            
            if self._MODEL_SIGNATURES['detection'](input_shapes, [None] * output_count):
                return 'detection'
            
            if len(input_shapes) > 0 and len(input_shapes[0]) >= 4:
                h, w = input_shapes[0][2], input_shapes[0][3]
                
                if h == 192 and w == 192:
                    return 'landmark_2d_106'
                
                if h == 96 and w == 96:
                    return 'genderage'
                
                if len(input_shapes) == 2 and h == 128 and w == 128:
                    return 'inswapper'
                
                if h == w and h >= 112 and h % 16 == 0:
                    return 'recognition'
            
            return None
            
        except Exception as e:
            print(f'Error identifying model {onnx_file}: {e}')
            return None
    
    def _load_model(self, task_name: str) -> Any:
        """Lazily load a model by task name.
        
        Args:
            task_name: Name of the task to load.
        
        Returns:
            Model instance or None if not found.
        """
        if task_name in self._loaded_models:
            return self.models.get(task_name)
        
        if task_name not in self._model_paths:
            return None
        
        onnx_file = self._model_paths[task_name]
        model = model_zoo.get_model(onnx_file, **self._init_kwargs)
        
        if model is not None:
            self.models[task_name] = model
            self._loaded_models.add(task_name)
            print(f'loaded model: {onnx_file}, task: {task_name}')
        
        return model
    
    def _ensure_det_model(self) -> Any:
        """Ensure detection model is loaded."""
        if self.det_model is None:
            self.det_model = self._load_model('detection')
        return self.det_model
    
    def prepare(
        self,
        ctx_id: int,
        det_thresh: float = 0.5,
        det_size: Tuple[int, int] = (640, 640),
    ) -> None:
        """Prepare models for inference.
        
        This method initializes the detection model with specified parameters.
        Other models are loaded lazily when needed.
        
        Args:
            ctx_id: Context ID for GPU device. Use -1 for CPU.
            det_thresh: Detection threshold for face detection. Defaults to 0.5.
            det_size: Input size for detection model. Defaults to (640, 640).
        """
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        
        self._ensure_det_model()
        if self.det_model is not None:
            self.det_model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
    
    def get(
        self,
        img: np.ndarray,
        max_num: int = 0,
        det_metric: str = 'default',
    ) -> List[Face]:
        """Detect and analyze faces in an image.
        
        Args:
            img: Input image as numpy array.
                - Format: BGR (OpenCV default) or RGB
                - Dtype: uint8
                - Range: 0-255
                - Shape: (H, W, 3)
            max_num: Maximum number of faces to detect. 0 means no limit.
            det_metric: Detection metric for face selection.
                - 'default': Prefer faces closer to image center
                - 'max': Select largest faces
        
        Returns:
            List of Face objects containing detection results and attributes.
            Each Face object may contain:
                - bbox: np.ndarray, shape (4,), bounding box [x1, y1, x2, y2]
                - kps: np.ndarray, shape (5, 2), 5 facial keypoints
                - det_score: float, detection confidence score
                - embedding: np.ndarray, face embedding vector (if recognition model loaded)
                - gender: int, 0 for female, 1 for male (if genderage model loaded)
                - age: int, estimated age (if genderage model loaded)
        """
        det_model = self._ensure_det_model()
        if det_model is None:
            return []
        
        bboxes, kpss = det_model.detect(img, max_num=max_num, metric=det_metric)
        
        if bboxes is None or bboxes.shape[0] == 0:
            return []
        
        num_faces = bboxes.shape[0]
        ret: List[Face] = []
        
        for i in range(num_faces):
            bbox = _safe_slice_bbox(bboxes, i)
            det_score = _safe_slice_score(bboxes, i)
            kps = _safe_slice_kps(kpss, i)
            
            if bbox is None:
                continue
            
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            
            for task_name in self._model_paths:
                if task_name == 'detection':
                    continue
                if self._allowed_modules is not None and task_name not in self._allowed_modules:
                    continue
                
                model = self._load_model(task_name)
                if model is not None:
                    try:
                        model.get(img, face)
                    except Exception as e:
                        print(f'Error running {task_name} model: {e}')
            
            ret.append(face)
        
        return ret
    
    def draw_on(
        self,
        img: np.ndarray,
        faces: List[Face],
    ) -> np.ndarray:
        """Draw face analysis results on image.
        
        Args:
            img: Input image as numpy array (BGR format, uint8, 0-255).
            faces: List of Face objects to visualize.
        
        Returns:
            Image with drawn bounding boxes, keypoints, and attributes.
        """
        return draw_faces(img, faces)
    
    def get_model(self, task_name: str) -> Optional[Any]:
        """Get a loaded model by task name.
        
        Args:
            task_name: Name of the task (e.g., 'detection', 'recognition').
        
        Returns:
            Model instance if loaded, None otherwise.
        """
        return self.models.get(task_name)
    
    def load_model(self, task_name: str) -> Optional[Any]:
        """Explicitly load a model by task name.
        
        Args:
            task_name: Name of the task to load.
        
        Returns:
            Loaded model instance.
        """
        return self._load_model(task_name)
    
    def unload_model(self, task_name: str) -> bool:
        """Unload a model to free memory.
        
        Args:
            task_name: Name of the task to unload.
        
        Returns:
            True if model was unloaded, False if not found.
        """
        if task_name in self.models:
            del self.models[task_name]
            self._loaded_models.discard(task_name)
            if task_name == 'detection':
                self.det_model = None
            return True
        return False
    
    def list_available_models(self) -> List[str]:
        """List all available model task names.
        
        Returns:
            List of task names for available models.
        """
        return list(self._model_paths.keys())
    
    def list_loaded_models(self) -> List[str]:
        """List currently loaded model task names.
        
        Returns:
            List of task names for loaded models.
        """
        return list(self._loaded_models)
