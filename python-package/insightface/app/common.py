from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.linalg import norm as l2norm


class Face:
    _ATTR_DEFS: Dict[str, Tuple[type, Any]] = {
        'bbox': (np.ndarray, None),
        'kps': (np.ndarray, None),
        'det_score': (float, None),
        'embedding': (np.ndarray, None),
        'gender': (int, None),
        'age': (int, None),
        'pose': (np.ndarray, None),
    }

    def __init__(self, d: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._data: Dict[str, Any] = {}
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            self._data[k] = v

    def __setattr__(self, name: str, value: Any) -> None:
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __getattr__(self, name: str) -> Any:
        if name == '_data':
            return super().__getattribute__(name)
        return self._data.get(name)

    def __setitem__(self, name: str, value: Any) -> None:
        self._data[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._data.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def items(self) -> List[Tuple[str, Any]]:
        return list(self._data.items())

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    @property
    def embedding_norm(self) -> Optional[float]:
        if self.embedding is None:
            return None
        return float(l2norm(self.embedding))

    @property
    def normed_embedding(self) -> Optional[np.ndarray]:
        if self.embedding is None:
            return None
        norm_val = self.embedding_norm
        if norm_val is None or norm_val == 0:
            return None
        return self.embedding / norm_val

    @property
    def sex(self) -> Optional[str]:
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'

    def get_bbox(self) -> Optional[np.ndarray]:
        bbox = self._data.get('bbox')
        if bbox is not None and len(bbox) >= 4:
            return bbox[:4]
        return None

    def get_kps(self) -> Optional[np.ndarray]:
        return self._data.get('kps')

    def get_det_score(self) -> Optional[float]:
        return self._data.get('det_score')

    def get_embedding(self) -> Optional[np.ndarray]:
        return self._data.get('embedding')

    def get_gender_age(self) -> Tuple[Optional[int], Optional[int]]:
        return self._data.get('gender'), self._data.get('age')

    def __repr__(self) -> str:
        attrs = []
        for k in ['bbox', 'det_score', 'kps', 'embedding', 'gender', 'age', 'pose']:
            v = self._data.get(k)
            if v is not None:
                if isinstance(v, np.ndarray):
                    attrs.append(f'{k}=array{v.shape}')
                else:
                    attrs.append(f'{k}={v}')
        return f"Face({', '.join(attrs)})"
