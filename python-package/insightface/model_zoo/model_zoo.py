# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

from __future__ import annotations

import os
import os.path as osp
import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import onnxruntime

from .arcface_onnx import ArcFaceONNX
from .retinaface import RetinaFace
from .landmark import Landmark
from .attribute import Attribute
from .inswapper import INSwapper
from ..utils import download_onnx

__all__ = ['get_model', 'PickableInferenceSession', 'ModelRouter']


class PickableInferenceSession(onnxruntime.InferenceSession):
    """A wrapper to make InferenceSession pickable for multiprocessing.
    
    This class extends onnxruntime.InferenceSession to support serialization
    via pickle, enabling use in multiprocessing scenarios.
    """
    
    def __init__(self, model_path: str, **kwargs: Any) -> None:
        """Initialize the pickable inference session.
        
        Args:
            model_path: Path to the ONNX model file.
            **kwargs: Additional arguments passed to InferenceSession.
        """
        super().__init__(model_path, **kwargs)
        self.model_path = model_path

    def __getstate__(self) -> Dict[str, str]:
        """Return state for pickling."""
        return {'model_path': self.model_path}

    def __setstate__(self, values: Dict[str, str]) -> None:
        """Restore state from pickle."""
        model_path = values['model_path']
        self.__init__(model_path)


class ModelRouter:
    """Router for identifying and instantiating appropriate model class.
    
    This class examines ONNX model structure to determine the appropriate
    model class (detection, recognition, landmark, etc.) and creates
    the corresponding model instance.
    """
    
    def __init__(self, onnx_file: str) -> None:
        """Initialize the model router.
        
        Args:
            onnx_file: Path to the ONNX model file.
        """
        self.onnx_file = onnx_file

    def get_model(self, **kwargs: Any) -> Optional[Union[ArcFaceONNX, RetinaFace, Landmark, Attribute, INSwapper]]:
        """Create and return the appropriate model instance.
        
        The model type is determined by examining the model's input/output shapes:
        - Detection models: >= 5 outputs
        - Landmark models: 192x192 input
        - Attribute models: 96x96 input
        - INSwapper: 2 inputs, 128x128 input
        - Recognition models: square input >= 112, divisible by 16
        
        Args:
            **kwargs: Arguments passed to model initialization, including:
                - providers: List of execution providers
                - provider_options: Provider-specific options
        
        Returns:
            Model instance of appropriate type, or None if unrecognized.
        """
        session = PickableInferenceSession(self.onnx_file, **kwargs)
        print(f'Applied providers: {session._providers}, with options: {session._provider_options}')
        inputs = session.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()

        if len(outputs) >= 5:
            return RetinaFace(model_file=self.onnx_file, session=session)
        elif input_shape[2] == 192 and input_shape[3] == 192:
            return Landmark(model_file=self.onnx_file, session=session)
        elif input_shape[2] == 96 and input_shape[3] == 96:
            return Attribute(model_file=self.onnx_file, session=session)
        elif len(inputs) == 2 and input_shape[2] == 128 and input_shape[3] == 128:
            return INSwapper(model_file=self.onnx_file, session=session)
        elif input_shape[2] == input_shape[3] and input_shape[2] >= 112 and input_shape[2] % 16 == 0:
            return ArcFaceONNX(model_file=self.onnx_file, session=session)
        else:
            return None


def find_onnx_file(dir_path: str) -> Optional[str]:
    """Find the most recent ONNX file in a directory.
    
    Args:
        dir_path: Path to directory to search.
    
    Returns:
        Path to the most recently modified ONNX file, or None if not found.
    """
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob("%s/*.onnx" % dir_path)
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]


def get_default_providers() -> List[str]:
    """Get default ONNX Runtime execution providers.
    
    Returns:
        List of provider names in priority order.
    """
    return ['CUDAExecutionProvider', 'CPUExecutionProvider']


def get_default_provider_options() -> Optional[Dict[str, Any]]:
    """Get default provider options.
    
    Returns:
        Provider options dictionary or None for defaults.
    """
    return None


def get_model(
    name: str,
    **kwargs: Any,
) -> Optional[Union[ArcFaceONNX, RetinaFace, Landmark, Attribute, INSwapper]]:
    """Load an ONNX model by name or path.
    
    Args:
        name: Model name or path to ONNX file.
            - If not ending with '.onnx', treated as model pack name
            - If ending with '.onnx', treated as direct file path
        **kwargs: Additional arguments including:
            - root: Root directory for model storage (default: '~/.insightface')
            - download: Whether to download if not found (default: False)
            - download_zip: Whether to download as zip (default: False)
            - providers: List of execution providers
            - provider_options: Provider-specific options
    
    Returns:
        Model instance of appropriate type, or None if not found.
    
    Example:
        >>> model = get_model('buffalo_l', download=True)
        >>> model = get_model('/path/to/model.onnx')
    """
    root = kwargs.get('root', '~/.insightface')
    root = os.path.expanduser(root)
    model_root = osp.join(root, 'models')
    allow_download = kwargs.get('download', False)
    download_zip = kwargs.get('download_zip', False)
    
    if not name.endswith('.onnx'):
        model_dir = os.path.join(model_root, name)
        model_file = find_onnx_file(model_dir)
        if model_file is None:
            return None
    else:
        model_file = name
    
    if not osp.exists(model_file) and allow_download:
        model_file = download_onnx('models', model_file, root=root, download_zip=download_zip)
    
    assert osp.exists(model_file), 'model_file %s should exist' % model_file
    assert osp.isfile(model_file), 'model_file %s should be a file' % model_file
    
    router = ModelRouter(model_file)
    providers = kwargs.get('providers', get_default_providers())
    provider_options = kwargs.get('provider_options', get_default_provider_options())
    model = router.get_model(providers=providers, provider_options=provider_options)
    return model
