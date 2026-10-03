"""Single-image values. No identity history, clocks or storage metadata."""
from dataclasses import dataclass, field, fields

import numpy as np


def validate_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) == 0):
        raise ValueError('image must be a nonempty HWC BGR uint8 array')
    return image


def normalize_feature(feature):
    value = np.asarray(feature)
    if value.ndim != 1 or not value.size or value.dtype.kind not in 'iuf':
        raise ValueError('feature must be a finite, nonempty numeric vector')
    value = value.astype(np.float32, copy=True)
    if not np.isfinite(value).all():
        raise ValueError('feature must be a finite, nonempty vector')
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError('feature must have nonzero finite norm')
    return np.ascontiguousarray(value / norm)


def _plain(value):
    if isinstance(value, Value):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


class Value:
    def to_dict(self):
        return {item.name: _plain(getattr(self, item.name)) for item in fields(self)
                if not item.name.startswith('_')}


@dataclass
class Face(Value):
    bbox: np.ndarray
    det_score: float
    kps: np.ndarray = None
    embedding: np.ndarray = None
    _reason: str = field(default=None, init=False, repr=False)


@dataclass
class Person(Value):
    body_bbox: np.ndarray = None
    det_score: float = None
    reid_feature: np.ndarray = None
    face: Face = None
    _face_model_id: str = field(default=None, init=False, repr=False)
    _reid_model_id: str = field(default=None, init=False, repr=False)
    _association_valid: bool = field(default=False, init=False, repr=False)


@dataclass(frozen=True)
class MatchResult(Value):
    observation: Person
    person_id: object = None
    matched_by: str = None
    similarity: float = None
    _owner: object = field(default=None, init=False, repr=False, compare=False)
    _version: int = field(default=-1, init=False, repr=False, compare=False)
    _body_feature: object = field(default=None, init=False, repr=False, compare=False)
    _used: bool = field(default=False, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class UpdateResult(Value):
    added: int = 0
    replaced: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class RegistrationResult(Value):
    person_id: object
    accepted: int
    rejected: list
