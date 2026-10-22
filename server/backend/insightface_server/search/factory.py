from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import SEARCH_PROFILES, MutableSearchIndex
from .native import DIMENSION, NativeSearchLibrary
from .reference import ReferenceSearchIndex


class ReferenceSearchBackend:
    name = "reference"

    def create_index(
        self, *, profile: str, dimension: int, capacity_rows: int
    ) -> MutableSearchIndex:
        return ReferenceSearchIndex(
            profile=profile, dimension=dimension, capacity_rows=capacity_rows
        )

    def runtime_summary(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "implementation": "NumPy exact grouped Flat-IP reference",
            "production": False,
            "profiles": sorted(SEARCH_PROFILES),
            "grouped_person_topk": True,
            "grouped_person_topk_mode": "host_reference",
        }

    def close(self) -> None:
        return None


class NativeSearchBackend:
    def __init__(
        self,
        path: Path,
        *,
        backend: str,
        device: int,
        topk_mode: str,
    ) -> None:
        self.name = backend
        self.topk_mode = topk_mode
        self.library = NativeSearchLibrary(path, backend=backend, device=device)
        self._self_test()

    def _self_test(self) -> None:
        import numpy as np

        index = self.library.create_index(
            profile="fp32_v1", capacity_rows=1, topk_mode=self.topk_mode
        )
        try:
            from .base import IndexRecord

            vector = np.full(DIMENSION, 1.0 / np.sqrt(DIMENSION), dtype=np.float32)
            index.add_batch([IndexRecord(1, 1, vector)])
            hits = index.search_persons(vector, 1)
            if len(hits) != 1 or hits[0].vector_id != 1 or abs(hits[0].cosine - 1.0) > 1e-4:
                raise RuntimeError("native grouped Top-K self-test returned an invalid result")
            if index.remove_batch([1]) != {1} or index.stats().live_rows != 0:
                raise RuntimeError("native mutation self-test failed")
        finally:
            index.close()

    def create_index(
        self, *, profile: str, dimension: int, capacity_rows: int
    ) -> MutableSearchIndex:
        if dimension != DIMENSION:
            raise ValueError(
                f"native search requires dimension {DIMENSION}; received {dimension}"
            )
        return self.library.create_index(
            profile=profile,
            capacity_rows=capacity_rows,
            topk_mode=self.topk_mode,
        )

    def runtime_summary(self) -> dict[str, Any]:
        capabilities = self.library.capabilities
        return {
            "backend": self.name,
            "profiles": list(capabilities.profiles),
            "device": capabilities.device,
            "compute_capability": capabilities.compute_capability,
            "cuda_runtime_version": capabilities.cuda_runtime_version,
            "cuda_driver_version": capabilities.cuda_driver_version,
            "build_info": capabilities.build_info,
            "topk_mode": self.topk_mode,
            "grouped_person_topk": True,
            "grouped_person_topk_mode": capabilities.grouped_person_topk_mode,
        }

    def close(self) -> None:
        return None


def create_search_backend(settings: Any) -> ReferenceSearchBackend | NativeSearchBackend:
    requested = str(getattr(settings, "search_backend", "auto"))
    inference_mode = str(getattr(settings, "inference_mode", "onnx"))
    provider = str(getattr(settings, "execution_provider", "CPUExecutionProvider"))
    if requested == "auto":
        if inference_mode == "mock":
            requested = "reference"
        elif provider == "CUDAExecutionProvider":
            requested = "native_cuda"
        else:
            requested = "native_cpu"
    if requested == "reference":
        return ReferenceSearchBackend()
    if requested not in {"native_cpu", "native_cuda"}:
        raise ValueError(f"unsupported search backend: {requested}")
    configured_path = getattr(settings, "search_library_path", None)
    filename = (
        "libifs_search_cuda.so" if requested == "native_cuda" else "libifs_search_cpu.so"
    )
    path = Path(configured_path) if configured_path else Path(
        "/opt/insightface/server/native/lib"
    ) / filename
    return NativeSearchBackend(
        path,
        backend=requested,
        device=int(getattr(settings, "search_device_id", 0)),
        topk_mode=str(getattr(settings, "search_topk_mode", "auto")),
    )
