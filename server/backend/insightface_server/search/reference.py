from __future__ import annotations

import numpy as np

from .base import (
    SEARCH_PROFILES,
    IndexRecord,
    IndexStats,
    PersonHit,
    SearchIndexCapacityError,
    SearchIndexStateError,
)
from .synchronization import ReadWriteLock

INT8_PROFILE_SCALES = {
    "int8_x1000_v1": 1000,
    "int8_x736_v1": 736,
}


def _validated_embedding(value: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (dimension,):
        raise ValueError(f"embedding must have shape ({dimension},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 2e-4:
        raise ValueError(f"embedding must be L2-normalized; norm={norm:.8f}")
    return np.ascontiguousarray(vector)


def _round_half_away_from_zero(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0, np.floor(value + 0.5), np.ceil(value - 0.5))


def quantize_int8(value: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError("INT8 scale must be positive")
    scaled = _round_half_away_from_zero(
        np.asarray(value, dtype=np.float32) * np.float32(scale)
    )
    return np.clip(scaled, -128, 127).astype(np.int8)


def quantize_int8_x1000(value: np.ndarray) -> np.ndarray:
    return quantize_int8(value, 1000)


def quantize_int8_x736(value: np.ndarray) -> np.ndarray:
    return quantize_int8(value, 736)


def quantize_bf16_to_fp32(value: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 (nearest-even) while retaining a NumPy FP32 container."""

    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    bits = contiguous.view(np.uint32).copy()
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bits = (bits + rounding) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def profile_similarity(profile: str, left: np.ndarray, right: np.ndarray) -> float:
    """Return the profile's unclipped raw similarity for two normalized vectors.

    This is the scalar semantic reference used by enrollment review. It applies
    the same storage/query conversion and accumulator scale as the selected
    search profile, but deliberately does not clamp or round the returned score.
    """

    if profile not in SEARCH_PROFILES:
        raise ValueError(f"unsupported search profile: {profile}")
    left_value = np.asarray(left, dtype=np.float32).reshape(-1)
    right_value = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_value.size == 0 or left_value.shape != right_value.shape:
        raise ValueError("embeddings must have the same non-empty shape")
    left_value = _validated_embedding(left_value, left_value.size)
    right_value = _validated_embedding(right_value, right_value.size)

    scale = INT8_PROFILE_SCALES.get(profile)
    if scale is not None:
        left_int = quantize_int8(left_value, scale).astype(np.int32)
        right_int = quantize_int8(right_value, scale).astype(np.int32)
        accumulator = int(np.dot(left_int, right_int))
        return accumulator / float(scale * scale)
    if profile == "fp16_v1":
        left_value = left_value.astype(np.float16).astype(np.float32)
        right_value = right_value.astype(np.float16).astype(np.float32)
    elif profile == "bf16_v1":
        left_value = quantize_bf16_to_fp32(left_value)
        right_value = quantize_bf16_to_fp32(right_value)
    return float(np.dot(left_value, right_value))


class ReferenceSearchIndex:
    """Deterministic exact grouped Flat-IP reference used by tests and shadowing.

    This implementation deliberately favors a small, auditable contract over
    throughput. Production images use the native CPU/CUDA implementations.
    """

    def __init__(self, *, profile: str, dimension: int, capacity_rows: int) -> None:
        if profile not in SEARCH_PROFILES:
            raise ValueError(f"unsupported search profile: {profile}")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if capacity_rows <= 0:
            raise ValueError("capacity_rows must be positive")
        self.profile = profile
        self.dimension = dimension
        self._capacity = capacity_rows
        self._lock = ReadWriteLock()
        self._closed = False
        self._vector_ids = np.zeros(capacity_rows, dtype=np.uint64)
        self._person_ids = np.zeros(capacity_rows, dtype=np.uint64)
        self._alive = np.zeros(capacity_rows, dtype=np.bool_)
        if profile in INT8_PROFILE_SCALES:
            self._vectors = np.zeros((capacity_rows, dimension), dtype=np.int8)
        elif profile == "fp16_v1":
            self._vectors = np.zeros((capacity_rows, dimension), dtype=np.float16)
        else:
            self._vectors = np.zeros((capacity_rows, dimension), dtype=np.float32)
        self._id_to_slot: dict[int, int] = {}
        self._free_slots: list[int] = []
        self._physical_rows = 0

    def _ensure_open(self) -> None:
        if self._closed:
            raise SearchIndexStateError("search index is closed")

    def _encode(self, value: np.ndarray) -> np.ndarray:
        int8_scale = INT8_PROFILE_SCALES.get(self.profile)
        if int8_scale is not None:
            return quantize_int8(value, int8_scale)
        if self.profile == "fp16_v1":
            return value.astype(np.float16)
        if self.profile == "bf16_v1":
            return quantize_bf16_to_fp32(value)
        return value.astype(np.float32, copy=False)

    def add_batch(self, records: list[IndexRecord]) -> None:
        if not records:
            return
        with self._lock.write():
            self._ensure_open()
            ids = [int(record.vector_id) for record in records]
            if any(value <= 0 or value > np.iinfo(np.int64).max for value in ids):
                raise ValueError("vector_id must be within 1..2^63-1")
            if any(int(record.person_numeric_id) <= 0 for record in records):
                raise ValueError("person_numeric_id must be positive")
            if len(ids) != len(set(ids)) or any(value in self._id_to_slot for value in ids):
                raise ValueError("active vector_id values must be unique")
            live_rows = len(self._id_to_slot)
            if live_rows + len(records) > self._capacity:
                raise SearchIndexCapacityError(
                    f"index capacity {self._capacity} would be exceeded"
                )
            encoded = [
                self._encode(_validated_embedding(record.embedding, self.dimension))
                for record in records
            ]
            available_new = self._capacity - self._physical_rows
            required_new = max(0, len(records) - len(self._free_slots))
            if required_new > available_new:
                raise SearchIndexCapacityError("no physical slots are available")
            for record, vector in zip(records, encoded, strict=True):
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    slot = self._physical_rows
                    self._physical_rows += 1
                vector_id = int(record.vector_id)
                self._vectors[slot] = vector
                self._vector_ids[slot] = vector_id
                self._person_ids[slot] = int(record.person_numeric_id)
                self._alive[slot] = True
                self._id_to_slot[vector_id] = slot

    def remove_batch(self, vector_ids: list[int]) -> set[int]:
        removed: set[int] = set()
        with self._lock.write():
            self._ensure_open()
            for value in vector_ids:
                vector_id = int(value)
                slot = self._id_to_slot.pop(vector_id, None)
                if slot is None:
                    continue
                self._alive[slot] = False
                self._free_slots.append(slot)
                removed.add(vector_id)
        return removed

    def _scores(self, query: np.ndarray, slots: np.ndarray) -> np.ndarray:
        int8_scale = INT8_PROFILE_SCALES.get(self.profile)
        if int8_scale is not None:
            encoded_query = quantize_int8(query, int8_scale).astype(np.int32)
            matrix = self._vectors[slots].astype(np.int32)
            values = matrix @ encoded_query
            return values.astype(np.float32) / np.float32(int8_scale * int8_scale)
        if self.profile == "fp16_v1":
            encoded_query = query.astype(np.float16).astype(np.float32)
            matrix = self._vectors[slots].astype(np.float32)
        elif self.profile == "bf16_v1":
            encoded_query = quantize_bf16_to_fp32(query)
            matrix = self._vectors[slots]
        else:
            encoded_query = query
            matrix = self._vectors[slots]
        return np.asarray(matrix @ encoded_query, dtype=np.float32)

    def search_persons(self, query: np.ndarray, limit: int) -> list[PersonHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        normalized = _validated_embedding(query, self.dimension)
        with self._lock.read():
            self._ensure_open()
            slots = np.flatnonzero(self._alive[: self._physical_rows])
            if slots.size == 0:
                return []
            scores = self._scores(normalized, slots)
            best: dict[int, tuple[float, int]] = {}
            for slot, raw_score in zip(slots.tolist(), scores.tolist(), strict=True):
                person_id = int(self._person_ids[slot])
                vector_id = int(self._vector_ids[slot])
                score = float(raw_score)
                current = best.get(person_id)
                if current is None or score > current[0] or (
                    score == current[0] and vector_id < current[1]
                ):
                    best[person_id] = (score, vector_id)
            ordered = [
                (person_id, value[1], value[0]) for person_id, value in best.items()
            ]
            ordered.sort(key=lambda hit: (-hit[2], hit[0], hit[1]))
            return [
                PersonHit(
                    person_numeric_id=person_id,
                    vector_id=vector_id,
                    cosine=float(np.clip(score, -1.0, 1.0)),
                )
                for person_id, vector_id, score in ordered[:limit]
            ]

    def stats(self) -> IndexStats:
        with self._lock.read():
            self._ensure_open()
            live = len(self._id_to_slot)
            return IndexStats(
                profile=self.profile,
                dimension=self.dimension,
                live_rows=live,
                physical_rows=self._physical_rows,
                capacity_rows=self._capacity,
                tombstone_rows=self._physical_rows - live,
            )

    def close(self) -> None:
        with self._lock.write():
            self._closed = True
            self._id_to_slot.clear()
            self._free_slots.clear()
