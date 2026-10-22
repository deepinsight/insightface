"""Typed, mapping-compatible API results."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast


class PixelBox(TypedDict):
    x: int
    y: int
    width: int
    height: int


class NormalizedBox(TypedDict):
    left: float
    top: float
    width: float
    height: float


class BoundingBox(TypedDict):
    pixels: PixelBox
    normalized: NormalizedBox


class Quality(TypedDict, total=False):
    score: float
    sharpness: float
    brightness: float


class _RequiredLivenessResult(TypedDict):
    status: Literal["ok", "input_rejected"]
    is_live: Optional[bool]
    live_score: Optional[float]


class LivenessResult(_RequiredLivenessResult, total=False):
    """Rejected input may include English retry guidance; older results omit it."""

    reason: str


class FaceObservation(TypedDict, total=False):
    id: str
    bbox: BoundingBox
    landmarks: List[List[float]]
    detection_score: float
    quality: Quality
    embedding: List[float]
    liveness: LivenessResult


class Person(TypedDict, total=False):
    id: str
    name: Optional[str]
    external_id: Optional[str]
    metadata: Dict[str, Any]
    face_count: int
    created_at: str
    updated_at: str


class Collection(TypedDict, total=False):
    id: str
    name: str
    description: str
    default_threshold: float
    model_id: str
    model_digest: str
    embedding_dimension: int
    preprocessing_version: str
    embedding_contract_id: str
    search_profile: str
    capacity_rows: int
    max_faces_per_person: int
    load_policy: str
    search_revision: int
    save_face_crops: bool
    metadata: Dict[str, Any]
    person_count: int
    face_count: int
    created_at: str
    updated_at: str
    detection: Dict[str, Any]
    detection_revision: int


class FaceSample(TypedDict, total=False):
    id: str
    person_id: str
    bounding_box: BoundingBox
    landmarks: List[List[float]]
    detection_score: float
    quality: Quality
    model_id: str
    model_digest: str
    preprocessing_version: str
    embedding_source: str
    embedding_contract_id: Optional[str]
    has_crop: bool
    created_at: str
    liveness: LivenessResult


class Match(TypedDict, total=False):
    person: Person
    similarity: float
    matched_face_id: str


class Monitor(TypedDict, total=False):
    id: str
    name: str
    description: str
    enabled: bool
    source: Dict[str, str]
    collection_id: str
    inference_fps: float
    match_threshold: Optional[float]
    event_buffer_size: int
    event_policy: Dict[str, Any]
    preview_enabled: bool
    runtime: Dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ApiResult(Mapping[str, Any]):
    """A successful response with request metadata.

    It behaves like a read-only mapping for callers that prefer raw JSON while
    specialized subclasses expose typed convenience properties.
    """

    data: Dict[str, Any]
    status_code: int
    request_id: Optional[str]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the response JSON."""

        return dict(self.data)


class DetectResult(ApiResult):
    @property
    def faces(self) -> List[FaceObservation]:
        return cast(List[FaceObservation], self.data.get("faces", []))

    @property
    def processing_ms(self) -> float:
        return float(self.data.get("processing_ms", 0.0))


class HealthResult(ApiResult):
    @property
    def status(self) -> str:
        return str(self.data.get("status", "unknown"))


class SystemResult(ApiResult):
    @property
    def execution_provider(self) -> Optional[str]:
        value = self.data.get("execution_provider")
        return str(value) if value is not None else None


class ModelsResult(ApiResult):
    @property
    def models(self) -> List[Dict[str, Any]]:
        return cast(List[Dict[str, Any]], self.data.get("models", []))

    @property
    def addons(self) -> List[Dict[str, Any]]:
        return cast(List[Dict[str, Any]], self.data.get("addons", []))


class CompareResult(ApiResult):
    @property
    def matched(self) -> bool:
        return bool(self.data.get("matched", False))

    @property
    def similarity(self) -> float:
        return float(self.data["similarity"])

    @property
    def threshold(self) -> float:
        return float(self.data["threshold"])


class EmbeddingsResult(ApiResult):
    @property
    def faces(self) -> List[FaceObservation]:
        # The API represents each embedding together with its detected face.
        return cast(List[FaceObservation], self.data.get("faces", []))


class CollectionResult(ApiResult):
    @property
    def collection(self) -> Collection:
        value = self.data.get("collection", self.data)
        return cast(Collection, value)


class CollectionPage(ApiResult):
    @property
    def collections(self) -> List[Collection]:
        return cast(List[Collection], self.data.get("collections", []))

    @property
    def next_cursor(self) -> Optional[str]:
        return cast(Optional[str], self.data.get("next_cursor"))


class MonitorResult(ApiResult):
    @property
    def monitor(self) -> Monitor:
        return cast(Monitor, self.data.get("monitor", self.data))


class MonitorPage(ApiResult):
    @property
    def monitors(self) -> List[Monitor]:
        return cast(List[Monitor], self.data.get("monitors", []))

    @property
    def next_cursor(self) -> Optional[str]:
        return cast(Optional[str], self.data.get("next_cursor"))


class MonitorStateResult(ApiResult):
    @property
    def state(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.data.get("state", self.data))


class MonitorEventPage(ApiResult):
    @property
    def events(self) -> List[Dict[str, Any]]:
        return cast(List[Dict[str, Any]], self.data.get("events", []))

    @property
    def next_cursor(self) -> Optional[str]:
        return cast(Optional[str], self.data.get("next_cursor"))

    @property
    def has_more(self) -> bool:
        return bool(self.data.get("has_more", False))

    @property
    def truncated(self) -> bool:
        return bool(self.data.get("truncated", False))

    @property
    def stream_reset(self) -> bool:
        return bool(self.data.get("stream_reset", False))


class PersonResult(ApiResult):
    @property
    def person(self) -> Person:
        return cast(Person, self.data.get("person", self.data))


class PersonRegistrationResult(PersonResult):
    @property
    def faces(self) -> List[FaceSample]:
        return cast(List[FaceSample], self.data.get("faces", []))

    @property
    def rejected_images(self) -> List[Dict[str, Any]]:
        return cast(List[Dict[str, Any]], self.data.get("rejected_images", []))


class FaceRegistrationResult(ApiResult):
    @property
    def faces(self) -> List[FaceSample]:
        return cast(List[FaceSample], self.data.get("faces", []))

    @property
    def rejected_images(self) -> List[Dict[str, Any]]:
        return cast(List[Dict[str, Any]], self.data.get("rejected_images", []))


class PersonPage(ApiResult):
    @property
    def persons(self) -> List[Person]:
        return cast(List[Person], self.data.get("persons", []))

    @property
    def next_cursor(self) -> Optional[str]:
        return cast(Optional[str], self.data.get("next_cursor"))


class FacePage(ApiResult):
    @property
    def faces(self) -> List[FaceSample]:
        return cast(List[FaceSample], self.data.get("faces", []))

    @property
    def next_cursor(self) -> Optional[str]:
        return cast(Optional[str], self.data.get("next_cursor"))


class SearchResult(ApiResult):
    @property
    def matches(self) -> List[Match]:
        return cast(List[Match], self.data.get("matches", []))

    @property
    def threshold(self) -> float:
        return float(self.data["threshold"])
