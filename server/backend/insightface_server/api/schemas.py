from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import normalize_detector_input_sizes

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_id(value: str) -> str:
    if ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "must be 1-64 characters starting with a letter or digit and containing "
            "only letters, digits, dot, underscore, or hyphen"
        )
    return value


def validate_collection_id(value: str) -> str:
    if value == "_default":
        return value
    return validate_id(value)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SearchProfile = Literal["fp32_v1", "fp16_v1", "bf16_v1", "int8_x736_v1", "int8_x1000_v1"]
SearchLoadPolicy = Literal["eager", "lazy"]
ReviewMode = Literal["off", "standard", "strict"]
EmbeddingMode = Literal["server", "external_trusted"]
SingleFaceSelection = Literal["largest", "center_largest"]


class CollectionDetectionCreate(StrictModel):
    input_sizes: list[tuple[int, int]] | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    nms_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    single_face_selection: SingleFaceSelection | None = None

    @field_validator("input_sizes")
    @classmethod
    def validate_input_sizes(
        cls, value: list[tuple[int, int]] | None
    ) -> list[tuple[int, int]] | None:
        if value is None:
            return None
        return list(normalize_detector_input_sizes(value))


class CollectionDetectionPatch(StrictModel):
    input_sizes: list[tuple[int, int]] | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    nms_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    single_face_selection: SingleFaceSelection | None = None

    @field_validator("input_sizes")
    @classmethod
    def validate_input_sizes(
        cls, value: list[tuple[int, int]] | None
    ) -> list[tuple[int, int]] | None:
        if value is None:
            return None
        return list(normalize_detector_input_sizes(value))

    @model_validator(mode="after")
    def reject_nulls(self) -> CollectionDetectionPatch:
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self


class CollectionSearchCreate(StrictModel):
    profile: SearchProfile | None = None
    capacity_rows: int | None = Field(default=None, ge=1, le=2**63 - 1)
    max_faces_per_person: int | None = Field(default=None, ge=1, le=1_000_000)
    load_policy: SearchLoadPolicy | None = None


class CollectionSearchPatch(StrictModel):
    capacity_rows: int | None = Field(default=None, ge=1, le=2**63 - 1)
    max_faces_per_person: int | None = Field(default=None, ge=1, le=1_000_000)
    load_policy: SearchLoadPolicy | None = None

    @model_validator(mode="after")
    def reject_nulls(self) -> CollectionSearchPatch:
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self


class CollectionCreate(StrictModel):
    id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    save_face_crops: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    detection: CollectionDetectionCreate = Field(
        default_factory=CollectionDetectionCreate
    )
    search: CollectionSearchCreate = Field(default_factory=CollectionSearchCreate)

    _id = field_validator("id")(validate_collection_id)


class CollectionPatch(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    save_face_crops: bool | None = None
    metadata: dict[str, Any] | None = None
    detection: CollectionDetectionPatch | None = None
    search: CollectionSearchPatch | None = None

    @model_validator(mode="after")
    def reject_nulls(self) -> CollectionPatch:
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self


class PersonPatch(StrictModel):
    name: str | None = Field(default=None, max_length=200)
    external_id: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def reject_null_metadata(self) -> PersonPatch:
        if "metadata" in self.model_fields_set and self.metadata is None:
            raise ValueError("metadata cannot be null")
        return self


class MonitorSource(StrictModel):
    type: Literal["rtsp"] = "rtsp"
    url: str = Field(min_length=8, max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_rtsp_url(cls, value: str) -> str:
        normalized = value.strip()
        parsed = urlsplit(normalized)
        if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
            raise ValueError("url must use the rtsp:// or rtsps:// scheme")
        if not parsed.hostname:
            raise ValueError("url must include a host")
        if parsed.fragment:
            raise ValueError("url must not include a fragment")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("url contains an invalid port") from exc
        return normalized


class MonitorEventPolicy(StrictModel):
    confirm_frames: int = Field(default=3, ge=1, le=100)
    absence_timeout_seconds: float = Field(default=3.0, ge=0.1, le=300.0)
    cooldown_seconds: float = Field(default=10.0, ge=0.0, le=3600.0)
    emit_unknown: bool = True


class MonitorEventPolicyPatch(StrictModel):
    confirm_frames: int | None = Field(default=None, ge=1, le=100)
    absence_timeout_seconds: float | None = Field(default=None, ge=0.1, le=300.0)
    cooldown_seconds: float | None = Field(default=None, ge=0.0, le=3600.0)
    emit_unknown: bool | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> MonitorEventPolicyPatch:
        if not self.model_fields_set:
            raise ValueError("at least one event policy field must be supplied")
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self


class MonitorCreate(StrictModel):
    id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    enabled: bool = True
    source: MonitorSource
    collection_id: str
    inference_fps: float = Field(default=2.0, ge=0.1, le=30.0)
    match_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    event_buffer_size: int = Field(default=1000, ge=10, le=10000)
    event_policy: MonitorEventPolicy = Field(default_factory=MonitorEventPolicy)
    preview_enabled: bool = False

    _id = field_validator("id")(validate_id)
    _collection_id = field_validator("collection_id")(validate_collection_id)


class MonitorPatch(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    enabled: bool | None = None
    source: MonitorSource | None = None
    collection_id: str | None = None
    inference_fps: float | None = Field(default=None, ge=0.1, le=30.0)
    match_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    event_buffer_size: int | None = Field(default=None, ge=10, le=10000)
    event_policy: MonitorEventPolicyPatch | None = None
    preview_enabled: bool | None = None

    @field_validator("collection_id")
    @classmethod
    def validate_optional_collection_id(cls, value: str | None) -> str | None:
        return validate_collection_id(value) if value is not None else None

    @model_validator(mode="after")
    def validate_patch(self) -> MonitorPatch:
        if not self.model_fields_set:
            raise ValueError("at least one field must be supplied")
        for name in self.model_fields_set:
            if name == "match_threshold":
                continue
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self
