from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import (
    IndexRecord,
    IndexStats,
    PersonHit,
    SearchIndexCapacityError,
    SearchIndexError,
    SearchIndexStateError,
)
from .synchronization import ReadWriteLock

ABI_VERSION = 2
DIMENSION = 512

_PROFILE_CODES = {
    "fp32_v1": 0,
    "fp16_v1": 1,
    "bf16_v1": 2,
    "int8_x1000_v1": 3,
    "int8_x736_v1": 4,
}
_BACKEND_CODES = {"native_cpu": 1, "native_cuda": 2}
_TOPK_CODES = {"auto": 0, "host": 1, "device": 2}

_OK = 0
_OUT_OF_MEMORY = 2
_UNSUPPORTED = 3
_CAPACITY_EXCEEDED = 6
_CAP_GROUPED_PERSON_TOPK = 1 << 5
_CAP_GROUPED_HOST_REFERENCE = 1 << 8
_CAP_GROUPED_DEVICE_RESIDENT = 1 << 9


class _Capabilities(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("dimension", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("profile_mask", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("device_topk_limit", ctypes.c_uint64),
        ("device", ctypes.c_int32),
        ("compute_capability_major", ctypes.c_int32),
        ("compute_capability_minor", ctypes.c_int32),
        ("cuda_runtime_version", ctypes.c_int32),
        ("cuda_driver_version", ctypes.c_int32),
    ]


class _CreateOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("reserve_rows", ctypes.c_uint64),
        ("max_rows", ctypes.c_uint64),
        ("device", ctypes.c_int32),
        ("topk_mode", ctypes.c_uint32),
        ("growth_factor", ctypes.c_double),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("physical_rows", ctypes.c_uint64),
        ("live_rows", ctypes.c_uint64),
        ("capacity_rows", ctypes.c_uint64),
        ("max_rows", ctypes.c_uint64),
        ("tombstone_rows", ctypes.c_uint64),
        ("reallocations", ctypes.c_uint64),
        ("bytes_per_vector", ctypes.c_uint64),
    ]


class _Timings(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("kernel_ms", ctypes.c_double),
        ("topk_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
    ]


@dataclass(frozen=True, slots=True)
class NativeCapabilities:
    backend: str
    profiles: tuple[str, ...]
    flags: int
    device: int
    compute_capability: str | None
    cuda_runtime_version: int
    cuda_driver_version: int
    build_info: str

    @property
    def grouped_person_topk_mode(self) -> str:
        if self.flags & _CAP_GROUPED_DEVICE_RESIDENT:
            return "device_exact"
        if self.flags & _CAP_GROUPED_HOST_REFERENCE:
            return "host_reference"
        return "unavailable"


def _decode(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else ""


def _validated_vectors(records: list[IndexRecord]) -> np.ndarray:
    values = np.asarray([record.embedding for record in records], dtype=np.float32)
    if values.shape != (len(records), DIMENSION):
        raise ValueError(f"native search requires vectors shaped (N, {DIMENSION})")
    if not np.all(np.isfinite(values)):
        raise ValueError("embedding contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(np.abs(norms - 1.0) > 2e-4):
        raise ValueError("all embeddings must be L2-normalized")
    return np.ascontiguousarray(values)


def _validated_query(query: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
    if value.shape != (DIMENSION,):
        raise ValueError(f"native search query must contain {DIMENSION} values")
    if not np.all(np.isfinite(value)):
        raise ValueError("query contains non-finite values")
    norm = float(np.linalg.norm(value))
    if abs(norm - 1.0) > 2e-4:
        raise ValueError("query must be L2-normalized")
    return value


class NativeSearchLibrary:
    """Validated process-wide handle for one CPU or CUDA ABI-v2 library."""

    def __init__(self, path: Path, *, backend: str, device: int) -> None:
        if backend not in _BACKEND_CODES:
            raise ValueError(f"unsupported native backend: {backend}")
        self.path = Path(path)
        self.backend = backend
        self.device = -1 if backend == "native_cpu" else int(device)
        try:
            self._library = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise SearchIndexError(f"unable to load native search library: {exc}") from exc
        self._bind()
        actual_abi = int(self._library.ifs_search_abi_version())
        if actual_abi != ABI_VERSION:
            raise SearchIndexError(
                f"native search ABI mismatch: expected {ABI_VERSION}, got {actual_abi}"
            )
        dimension = int(self._library.ifs_search_dimension())
        if dimension != DIMENSION:
            raise SearchIndexError(
                f"native search dimension mismatch: expected {DIMENSION}, got {dimension}"
            )
        raw = _Capabilities()
        raw.struct_size = ctypes.sizeof(_Capabilities)
        self._check(self._library.ifs_search_get_capabilities(self.device, ctypes.byref(raw)))
        expected_backend = _BACKEND_CODES[backend]
        if raw.backend != expected_backend:
            raise SearchIndexError(
                f"native library backend mismatch: expected {expected_backend}, got {raw.backend}"
            )
        if raw.abi_version != ABI_VERSION or raw.dimension != DIMENSION:
            raise SearchIndexError("native capability metadata is inconsistent")
        if not raw.flags & _CAP_GROUPED_PERSON_TOPK:
            raise SearchIndexError("native library lacks exact grouped Person Top-K")
        supported = tuple(
            name for name, code in _PROFILE_CODES.items() if raw.profile_mask & (1 << code)
        )
        capability = None
        if backend == "native_cuda" and raw.compute_capability_major >= 0:
            capability = f"{raw.compute_capability_major}.{raw.compute_capability_minor}"
        self.capabilities = NativeCapabilities(
            backend=backend,
            profiles=supported,
            flags=int(raw.flags),
            device=int(raw.device),
            compute_capability=capability,
            cuda_runtime_version=int(raw.cuda_runtime_version),
            cuda_driver_version=int(raw.cuda_driver_version),
            build_info=_decode(self._library.ifs_search_build_info()),
        )

    def _bind(self) -> None:
        library = self._library
        library.ifs_search_abi_version.restype = ctypes.c_uint32
        library.ifs_search_dimension.restype = ctypes.c_uint32
        library.ifs_search_build_info.restype = ctypes.c_char_p
        library.ifs_search_last_error.restype = ctypes.c_char_p
        library.ifs_search_status_string.argtypes = [ctypes.c_int]
        library.ifs_search_status_string.restype = ctypes.c_char_p
        library.ifs_search_get_capabilities.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(_Capabilities),
        ]
        library.ifs_search_get_capabilities.restype = ctypes.c_int
        library.ifs_search_create.argtypes = [
            ctypes.POINTER(_CreateOptions),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.ifs_search_create.restype = ctypes.c_int
        library.ifs_search_destroy.argtypes = [ctypes.c_void_p]
        library.ifs_search_destroy.restype = None
        library.ifs_search_add_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
        ]
        library.ifs_search_add_batch.restype = ctypes.c_int
        library.ifs_search_delete_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        library.ifs_search_delete_batch.restype = ctypes.c_int
        library.ifs_search_grouped_topk.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(_Timings),
        ]
        library.ifs_search_grouped_topk.restype = ctypes.c_int
        library.ifs_search_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Stats)]
        library.ifs_search_get_stats.restype = ctypes.c_int

    def _check(self, status: int) -> None:
        if status == _OK:
            return
        detail = _decode(self._library.ifs_search_last_error())
        label = _decode(self._library.ifs_search_status_string(status)) or str(status)
        message = f"native search {label}: {detail or 'no detail'}"
        if status == _CAPACITY_EXCEEDED:
            raise SearchIndexCapacityError(message)
        if status in {_OUT_OF_MEMORY, _UNSUPPORTED}:
            raise SearchIndexError(message)
        raise SearchIndexError(message)

    def create_index(
        self,
        *,
        profile: str,
        capacity_rows: int,
        topk_mode: str = "auto",
    ) -> NativeSearchIndex:
        if profile not in _PROFILE_CODES:
            raise ValueError(f"unsupported search profile: {profile}")
        if profile not in self.capabilities.profiles:
            raise SearchIndexError(
                f"{self.backend} does not support search profile {profile}"
            )
        if topk_mode not in _TOPK_CODES:
            raise ValueError(f"unsupported Top-K mode: {topk_mode}")
        if capacity_rows <= 0:
            raise ValueError("capacity_rows must be positive")
        options = _CreateOptions()
        options.struct_size = ctypes.sizeof(_CreateOptions)
        options.profile = _PROFILE_CODES[profile]
        options.reserve_rows = capacity_rows
        options.max_rows = capacity_rows
        options.device = self.device
        options.topk_mode = _TOPK_CODES[topk_mode]
        options.growth_factor = 1.5
        handle = ctypes.c_void_p()
        self._check(self._library.ifs_search_create(ctypes.byref(options), ctypes.byref(handle)))
        if not handle.value:
            raise SearchIndexError("native search returned a null index handle")
        return NativeSearchIndex(self, handle, profile=profile)


class NativeSearchIndex:
    dimension = DIMENSION

    def __init__(
        self,
        library: NativeSearchLibrary,
        handle: ctypes.c_void_p,
        *,
        profile: str,
    ) -> None:
        self.library = library
        self.profile = profile
        self._handle = handle
        self._lock = ReadWriteLock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed or not self._handle.value:
            raise SearchIndexStateError("native search index is closed")

    def add_batch(self, records: list[IndexRecord]) -> None:
        if not records:
            return
        values = _validated_vectors(records)
        vector_ids = np.ascontiguousarray(
            [int(record.vector_id) for record in records], dtype=np.uint64
        )
        group_ids = np.ascontiguousarray(
            [int(record.person_numeric_id) for record in records], dtype=np.uint64
        )
        if np.any(vector_ids == 0) or np.any(group_ids == 0):
            raise ValueError("native vector and Person IDs must be positive")
        with self._lock.write():
            self._ensure_open()
            self.library._check(
                self.library._library.ifs_search_add_batch(
                    self._handle,
                    vector_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    group_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    len(records),
                )
            )

    def remove_batch(self, vector_ids: list[int]) -> set[int]:
        if not vector_ids:
            return set()
        values = np.ascontiguousarray(vector_ids, dtype=np.uint64)
        if np.any(values == 0):
            raise ValueError("native vector IDs must be positive")
        removed = ctypes.c_uint64()
        with self._lock.write():
            self._ensure_open()
            self.library._check(
                self.library._library.ifs_search_delete_batch(
                    self._handle,
                    values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    len(vector_ids),
                    ctypes.byref(removed),
                )
            )
        if removed.value != len(vector_ids):
            raise SearchIndexError(
                "native delete count mismatch; the collection index must be rebuilt"
            )
        return {int(value) for value in vector_ids}

    def search_persons(self, query: np.ndarray, limit: int) -> list[PersonHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        value = _validated_query(query)
        group_ids = np.empty(limit, dtype=np.uint64)
        vector_ids = np.empty(limit, dtype=np.uint64)
        scores = np.empty(limit, dtype=np.float32)
        count = ctypes.c_uint64()
        timings = _Timings()
        timings.struct_size = ctypes.sizeof(_Timings)
        with self._lock.read():
            self._ensure_open()
            self.library._check(
                self.library._library.ifs_search_grouped_topk(
                    self._handle,
                    value.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    limit,
                    group_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    vector_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.byref(count),
                    ctypes.byref(timings),
                )
            )
        return [
            PersonHit(
                person_numeric_id=int(group_ids[offset]),
                vector_id=int(vector_ids[offset]),
                cosine=float(np.clip(scores[offset], -1.0, 1.0)),
            )
            for offset in range(int(count.value))
        ]

    def stats(self) -> IndexStats:
        raw = _Stats()
        raw.struct_size = ctypes.sizeof(_Stats)
        with self._lock.read():
            self._ensure_open()
            self.library._check(
                self.library._library.ifs_search_get_stats(self._handle, ctypes.byref(raw))
            )
        return IndexStats(
            profile=self.profile,
            dimension=DIMENSION,
            live_rows=int(raw.live_rows),
            physical_rows=int(raw.physical_rows),
            capacity_rows=int(raw.max_rows or raw.capacity_rows),
            tombstone_rows=int(raw.tombstone_rows),
            reallocations=int(raw.reallocations),
        )

    def close(self) -> None:
        with self._lock.write():
            if self._closed:
                return
            handle = self._handle
            self._handle = ctypes.c_void_p()
            self._closed = True
            if handle.value:
                self.library._library.ifs_search_destroy(handle)
