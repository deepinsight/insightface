from __future__ import annotations

import pytest
from insightface_server.inference.onnx_engine import (
    audit_cuda_profile,
    validate_gpu_requirements,
)


def _event(provider: str, operator: str, **extra: object) -> dict[str, object]:
    args: dict[str, object] = {"provider": provider, "op_name": operator}
    args.update(extra)
    return {"cat": "Node", "name": f"{operator}_kernel", "args": args}


def test_cuda_profile_allows_only_small_integer_shape_metadata() -> None:
    profile = [
        _event("CUDAExecutionProvider", "Conv"),
        _event(
            "CPUExecutionProvider",
            "Gather",
            output_size="8",
            output_type_shape=[{"int64": [1]}],
        ),
    ]
    result = audit_cuda_profile(profile, model_name="detector")
    assert result["accepted"] is True
    assert result["cuda_kernel_count"] == 1
    assert result["cpu_shape_kernel_count"] == 1


@pytest.mark.parametrize(
    "cpu_event",
    [
        _event(
            "CPUExecutionProvider",
            "Conv",
            output_size="1024",
            output_type_shape=[{"float": [1, 16]}],
        ),
        _event(
            "CPUExecutionProvider",
            "Gather",
            output_size="8192",
            output_type_shape=[{"int64": [1024]}],
        ),
    ],
)
def test_cuda_profile_rejects_compute_or_large_cpu_fallback(
    cpu_event: dict[str, object],
) -> None:
    with pytest.raises(RuntimeError, match="rejected"):
        audit_cuda_profile(
            [_event("CUDAExecutionProvider", "Conv"), cpu_event],
            model_name="detector",
        )


def test_cuda_profile_requires_actual_cuda_kernel() -> None:
    with pytest.raises(RuntimeError, match="no CUDA kernels"):
        audit_cuda_profile([], model_name="recognizer")


def test_gpu_driver_and_blackwell_runtime_requirements() -> None:
    validate_gpu_requirements(
        [
            {
                "name": "RTX 5090",
                "compute_capability": "12.0",
                "driver_version": "580.105.08",
            }
        ],
        "12.9",
    )
    with pytest.raises(RuntimeError, match="570.26"):
        validate_gpu_requirements(
            [
                {
                    "name": "RTX 5090",
                    "compute_capability": "12.0",
                    "driver_version": "565.0",
                }
            ],
            "12.9",
        )
    with pytest.raises(RuntimeError, match="CUDA Runtime 12.8"):
        validate_gpu_requirements(
            [
                {
                    "name": "RTX 5090",
                    "compute_capability": "12.0",
                    "driver_version": "580.0",
                }
            ],
            "12.7",
        )
