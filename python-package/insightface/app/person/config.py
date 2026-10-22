"""Small, explicit options for single-image analysis and in-memory references."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class PersonConfig:
    face_similarity_threshold: float = .45
    reid_similarity_threshold: float = .85
    face_margin: float = .05
    reid_margin: float = .10
    face_min_size: int = 20
    face_registration_min_size: int = 32
    face_min_score: float = .6
    face_det_size: int = 0
    body_det_size: int = 0
    body_min_size: int = 16
    body_min_score: float = .5
    face_body_margin: float = .12
    max_body_samples: int = 4
    reference_capacity: int = 256
    duplicate_similarity_threshold: float = .98
    cpu_threads: int = 4

    def __post_init__(self):
        integers = {'face_min_size', 'face_registration_min_size', 'body_min_size',
                    'max_body_samples', 'reference_capacity', 'cpu_threads'}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in ('body_det_size', 'face_det_size'):
                multiple = 64 if field.name == 'body_det_size' else 32
                if type(value) is not int or value < 0 or value % multiple:
                    default = '640' if field.name == 'face_det_size' else 'model default'
                    raise ValueError(f'{field.name} must be 0 ({default}) or a positive multiple of {multiple}')
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{field.name} must be finite numeric')
            if field.name in integers:
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f'{field.name} must be a positive integer')
            elif not 0 <= value <= 1:
                raise ValueError(f'{field.name} must be between 0 and 1')
