"""Public response schemas for the versioned REST API.

The route implementations intentionally return ``JSONResponse`` objects so the
wire format remains explicit. These models document and test that format
without asking FastAPI to filter successful response payloads.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PublicResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ResponseEnvelope(PublicResponseModel):
    request_id: UUID


class ErrorBody(PublicResponseModel):
    code: str
    message: str
    details: dict[str, Any]


class ErrorEnvelope(ResponseEnvelope):
    error: ErrorBody


class PixelBox(PublicResponseModel):
    x: int
    y: int
    width: int
    height: int


class NormalizedBox(PublicResponseModel):
    left: float
    top: float
    width: float
    height: float


class BoundingBox(PublicResponseModel):
    pixels: PixelBox
    normalized: NormalizedBox


class FaceQuality(PublicResponseModel):
    score: float
    sharpness: float
    brightness: float
    pose: float


class LivenessResult(BaseModel):
    """Three core fields describe the evaluation; rejected input may also include reason.

    Live: {"status":"ok","is_live":true,"live_score":0.98}.
    Fake: {"status":"ok","is_live":false,"live_score":0.12}.
    Unsuitable input: {"status":"input_rejected","is_live":null,"live_score":null}.
    For input_rejected, optional reason is an English explanation with guidance for retrying.
    Older results may omit reason. Successful and fake evaluations have only the three core fields.
    """

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "input_rejected"] = Field(
        description=(
            "ok means the model evaluated the face, including fake verdicts. "
            "input_rejected means the input does not meet liveness requirements "
            "(for example, insufficient image coverage near an edge)."
        ),
    )
    is_live: bool | None = Field(
        description=(
            "For status=ok, true or false according to the configured liveness_threshold. "
            "Null only for input_rejected; it does not mean the addon is disabled."
        ),
    )
    live_score: float | None = Field(
        description="Live probability from 0 to 1 for status=ok; null for input_rejected.",
    )
    reason: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Optional English explanation and retry guidance, present only for input_rejected. "
            "Older results may omit it. Omitted for status=ok."
        ),
    )


class FaceObservation(PublicResponseModel):
    bbox: BoundingBox
    landmarks: list[list[float]]
    detection_score: float
    quality: FaceQuality
    embedding: list[float] | None = None
    liveness: LivenessResult | None = Field(
        default=None,
        description="Omitted when liveness was not evaluated for this face.",
    )


class ModelComponent(PublicResponseModel):
    model_id: str
    task: str
    sha256: str
    file: str | None = None
    input_size: list[int] | None = None
    embedding_dimension: int | None = None
    preprocessing_version: str | None = None


class ModelSummary(PublicResponseModel):
    model_id: str
    model_digest: str
    embedding_dimension: int
    preprocessing_version: str
    provider: str
    models: list[ModelComponent]
    license: dict[str, Any] | None
    addons: list[ModelComponent] | None = None


class DetectionProfile(PublicResponseModel):
    input_sizes: list[list[int]]
    threshold: float
    nms_threshold: float
    single_face_selection: Literal["largest", "center_largest"]


class Collection(PublicResponseModel):
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
    load_policy: Literal["eager", "lazy"]
    search_revision: int
    save_face_crops: bool
    metadata: dict[str, Any]
    person_count: int
    face_count: int
    created_at: str
    updated_at: str
    detection: DetectionProfile
    detection_revision: int


class Person(PublicResponseModel):
    id: str
    name: str | None
    external_id: str | None
    metadata: dict[str, Any]
    face_count: int
    created_at: str
    updated_at: str


class FaceSample(PublicResponseModel):
    id: str
    person_id: str | None = None
    bounding_box: BoundingBox
    landmarks: list[list[float]] | None
    detection_score: float
    quality: FaceQuality
    model_id: str
    model_digest: str
    preprocessing_version: str
    embedding_source: Literal["server", "external_trusted"]
    embedding_contract_id: str | None
    has_crop: bool
    created_at: str
    liveness: LivenessResult | None = Field(
        default=None,
        description=(
            "Saved evaluation from registration, omitted when no evaluation occurred. "
            "Historical results remain available even if liveness is now disabled."
        ),
    )


class RejectedImage(PublicResponseModel):
    index: int
    filename: str
    reason: str = Field(
        description=(
            "Actual rejection reason, such as low_quality, liveness_fake, or "
            "liveness_input_rejected. A separate liveness result does not replace this reason."
        ),
    )
    liveness: LivenessResult | None = Field(
        default=None,
        description=(
            "Supplemental evaluation when available, including rejection for other reasons. "
            "Omitted when liveness was not evaluated."
        ),
    )


class Match(PublicResponseModel):
    person: Person
    similarity: float
    matched_face_id: str


class MonitorSource(PublicResponseModel):
    type: Literal["rtsp"]
    url: str


class MonitorEventPolicy(PublicResponseModel):
    confirm_frames: int
    absence_timeout_seconds: float
    cooldown_seconds: float
    emit_unknown: bool


class MonitorRuntime(PublicResponseModel):
    status: str
    connected: bool
    stream_epoch: str | None
    last_frame_at: str | None
    last_error: dict[str, Any] | str | None
    preview_active: bool
    preview_viewers: int


class Monitor(PublicResponseModel):
    id: str
    name: str
    description: str
    enabled: bool
    source: MonitorSource
    collection_id: str
    inference_fps: float
    match_threshold: float | None
    event_buffer_size: int
    event_policy: MonitorEventPolicy
    preview_enabled: bool
    runtime: MonitorRuntime
    created_at: str
    updated_at: str


class MonitorStateSource(PublicResponseModel):
    width: int | None
    height: int | None
    source_fps: float | None


class MonitorInferenceState(PublicResponseModel):
    configured_fps: float
    actual_fps: float
    processing_ms: float | None
    last_frame_at: str | None
    last_inference_at: str | None
    frame_sequence: int
    decoded_frames: int
    processed_frames: int
    dropped_frames: int
    capacity_limited: bool


class MonitorPreviewState(PublicResponseModel):
    enabled: bool
    active: bool
    viewers: int
    frames: int


class MonitorState(PublicResponseModel):
    monitor_id: str
    stream_epoch: str | None
    status: str
    connected: bool
    source: MonitorStateSource
    inference: MonitorInferenceState
    threshold: float | None
    faces: list[dict[str, Any]]
    matched_faces: int
    unknown_faces: int
    liveness_blocked_faces: int = Field(
        default=0,
        description=(
            "Faces blocked by normal-mode liveness; counted separately from unknown_faces "
            "and excluded from identity search and person_enter events."
        ),
    )
    preview: MonitorPreviewState
    reconnects: int
    inference_errors: int
    last_error: dict[str, Any] | str | None
    started_at: str | None


class MonitorEvent(PublicResponseModel):
    id: UUID
    sequence: int
    type: str
    monitor_id: str
    stream_epoch: str
    occurred_at: str
    track_id: str | None = None
    person: dict[str, Any] | None = None
    similarity: float | None = None
    threshold: float | None = None
    matched_face_id: str | None = None
    face: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class HealthResponse(ResponseEnvelope):
    status: Literal["ready", "not_ready"]
    version: str
    auth_enabled: bool


class SystemResponse(ResponseEnvelope):
    server_version: str
    os: str
    architecture: str
    cpu: dict[str, Any]
    runtime: dict[str, Any]
    execution_provider: str
    model: ModelSummary
    database: dict[str, Any]
    data: dict[str, Any]
    models: dict[str, Any]
    stats: dict[str, int]
    search: dict[str, Any]
    api_key: dict[str, bool]
    safe_config: dict[str, Any]
    recent_errors: list[dict[str, Any]]


class ModelsResponse(ResponseEnvelope):
    models: list[ModelComponent]
    execution_provider: str
    license: dict[str, Any] | None
    addons: list[ModelComponent] | None = None


class AddonPreparationError(BaseModel):
    code: str
    message: str


class LivenessManagementResponse(ResponseEnvelope):
    enabled: bool = Field(description="Whether liveness is active in the running Server.")
    installed: bool = Field(description="Whether the local addon passes the official SHA256 check.")
    configured_enabled: bool = Field(description="Whether the current on-disk config enables liveness for the next startup.")
    restart_required: bool = Field(description="The saved liveness selection differs from the running Server.")
    can_enable: bool
    unavailable_code: str | None = Field(description="Stable reason code for clients to localize unavailable_reason; null when preparation is available.")
    unavailable_reason: str | None
    state: Literal["idle", "downloading", "ready", "error"]
    error: AddonPreparationError | None
    model_path: str
    config_file: str | None


class DetectResponse(ResponseEnvelope):
    faces: list[FaceObservation]
    processing_ms: float


class CompareResponse(ResponseEnvelope):
    matched: bool
    similarity: float
    threshold: float
    source_face: FaceObservation
    target_face: FaceObservation
    processing_ms: float


class EmbeddingsResponse(ResponseEnvelope):
    faces: list[FaceObservation]
    model: ModelSummary
    processing_ms: float


class CollectionResponse(ResponseEnvelope):
    collection: Collection


class CollectionPageResponse(ResponseEnvelope):
    collections: list[Collection]
    next_cursor: str | None


class PersonResponse(ResponseEnvelope):
    person: Person


class PersonRegistrationResponse(PersonResponse):
    faces: list[FaceSample]
    rejected_images: list[RejectedImage]


class PersonPageResponse(ResponseEnvelope):
    persons: list[Person]
    next_cursor: str | None


class FaceRegistrationResponse(ResponseEnvelope):
    faces: list[FaceSample]
    rejected_images: list[RejectedImage]


class FacePageResponse(ResponseEnvelope):
    faces: list[FaceSample]
    next_cursor: str | None


class SearchResponse(ResponseEnvelope):
    searched_face: FaceObservation
    matches: list[Match]
    threshold: float
    processing_ms: float


class MonitorResponse(ResponseEnvelope):
    monitor: Monitor


class MonitorPageResponse(ResponseEnvelope):
    monitors: list[Monitor]
    next_cursor: str | None


class MonitorStateResponse(ResponseEnvelope):
    state: MonitorState


class MonitorEventPageResponse(ResponseEnvelope):
    monitor_id: str
    stream_epoch: str | None
    events: list[MonitorEvent]
    oldest_sequence: int | None
    latest_sequence: int
    has_more: bool
    truncated: bool
    stream_reset: bool
    next_cursor: str
