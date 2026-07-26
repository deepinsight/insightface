from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

SEARCH_PROFILES = frozenset(
    {"fp32_v1", "fp16_v1", "bf16_v1", "int8_x1000_v1", "int8_x736_v1"}
)


class SearchIndexError(RuntimeError):
    """Base error raised by an in-process vector index."""


class SearchIndexCapacityError(SearchIndexError):
    """The configured hard row capacity would be exceeded."""


class SearchIndexStateError(SearchIndexError):
    """The index cannot serve the requested operation in its current state."""


@dataclass(frozen=True, slots=True)
class IndexRecord:
    vector_id: int
    person_numeric_id: int
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class PersonHit:
    person_numeric_id: int
    vector_id: int
    cosine: float


@dataclass(frozen=True, slots=True)
class IndexStats:
    profile: str
    dimension: int
    live_rows: int
    physical_rows: int
    capacity_rows: int
    tombstone_rows: int
    reallocations: int = 0


@runtime_checkable
class MutableSearchIndex(Protocol):
    profile: str
    dimension: int

    def add_batch(self, records: list[IndexRecord]) -> None: ...

    def remove_batch(self, vector_ids: list[int]) -> set[int]: ...

    def search_persons(self, query: np.ndarray, limit: int) -> list[PersonHit]: ...

    def stats(self) -> IndexStats: ...

    def close(self) -> None: ...


@runtime_checkable
class SearchBackend(Protocol):
    name: str

    def create_index(
        self, *, profile: str, dimension: int, capacity_rows: int
    ) -> MutableSearchIndex: ...

    def runtime_summary(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
