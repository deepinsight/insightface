"""Dataclasses used by the GUI core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FaceRecord:
    bbox: List[float]
    kps: Optional[List[List[float]]]
    det_score: float
    embedding: Optional[np.ndarray]
    normed_embedding: Optional[np.ndarray]
    face_id: Optional[str] = None
    gender: Optional[int] = None
    age: Optional[int] = None
    quality_score: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)
    crop: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    frame_index: Optional[int] = None
    timestamp_ms: Optional[int] = None

    def to_json_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "face_id": self.face_id,
            "bbox": self.bbox,
            "kps": self.kps,
            "det_score": self.det_score,
            "gender": self.gender,
            "age": self.age,
            "source_path": self.source_path,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
        }
        if include_embedding:
            data["embedding"] = self.embedding.tolist() if self.embedding is not None else None
            data["normed_embedding"] = (
                self.normed_embedding.tolist() if self.normed_embedding is not None else None
            )
        return data


@dataclass
class CompareResult:
    similarity: float
    threshold: float
    decision: str
    face_a: FaceRecord
    face_b: FaceRecord
    notes: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "similarity": self.similarity,
            "threshold": self.threshold,
            "decision": self.decision,
            "face_a": self.face_a.to_json_dict(),
            "face_b": self.face_b.to_json_dict(),
            "notes": self.notes,
        }


@dataclass
class SearchResult:
    person_id: Optional[int]
    person_name: str
    sample_id: Optional[int]
    similarity: float
    status: str
    crop_path: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "person_name": self.person_name,
            "sample_id": self.sample_id,
            "similarity": self.similarity,
            "status": self.status,
            "crop_path": self.crop_path,
        }


@dataclass
class EvaluationResult:
    scenario: str
    model_name: str
    provider: str
    threshold: float
    dataset_summary: Dict[str, Any]
    metrics: Dict[str, Any]
    errors: List[Dict[str, Any]]
    latency: Dict[str, Any]
    license_status: str
    created_at: str
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    threshold_recommendation: Optional[float] = None
    report_path: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "model_name": self.model_name,
            "provider": self.provider,
            "threshold": self.threshold,
            "dataset_summary": self.dataset_summary,
            "metrics": self.metrics,
            "errors": self.errors,
            "latency": self.latency,
            "license_status": self.license_status,
            "created_at": self.created_at,
            "raw_results": self.raw_results,
            "threshold_recommendation": self.threshold_recommendation,
            "report_path": self.report_path,
        }
