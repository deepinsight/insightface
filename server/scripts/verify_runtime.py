#!/usr/bin/env python3
"""Fail-fast verification for the pinned CPU and CUDA container runtimes."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import subprocess
import sys
from typing import Any

# This standalone check imports ORT without importing either application package.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"


class VerificationError(RuntimeError):
    pass


def version(value: str) -> tuple[int, ...]:
    parts = tuple(int(item) for item in re.findall(r"\d+", value))
    if not parts:
        raise VerificationError(f"Unable to parse version {value!r}")
    return parts


def cuda_version() -> str:
    try:
        library = ctypes.CDLL("libcudart.so.12")
        value = ctypes.c_int()
        status = int(library.cudaRuntimeGetVersion(ctypes.byref(value)))
    except (AttributeError, OSError) as exc:
        raise VerificationError(f"Unable to load CUDA 12 runtime: {exc}") from exc
    if status != 0 or value.value <= 0:
        raise VerificationError(f"cudaRuntimeGetVersion failed with status {status}")
    return f"{value.value // 1000}.{(value.value % 1000) // 10}.{value.value % 10}"


def cudnn_version() -> str:
    try:
        library = ctypes.CDLL("libcudnn.so.9")
        getter = library.cudnnGetVersion
        getter.restype = ctypes.c_size_t
        raw = int(getter())
    except (AttributeError, OSError) as exc:
        raise VerificationError(f"Unable to load cuDNN 9: {exc}") from exc
    if raw <= 0:
        raise VerificationError("cudnnGetVersion returned an invalid value")
    return f"{raw // 10000}.{(raw % 10000) // 100}.{raw % 100}"


def package_version(name: str) -> str:
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise VerificationError(f"Unable to query {name}: {exc}") from exc
    return result.stdout.strip()


def gpu_details() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise VerificationError(
            "No usable NVIDIA GPU was exposed by NVIDIA Container Toolkit"
        ) from exc
    values: list[dict[str, str]] = []
    supported = {(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise VerificationError(f"Unexpected nvidia-smi output: {line!r}")
        capability = version(fields[1])[:2]
        driver = version(fields[2])
        if capability not in supported:
            raise VerificationError(
                f"Unsupported Compute Capability {fields[1]} on {fields[0]}"
            )
        minimum = (570, 26) if capability >= (10, 0) else (535,)
        if driver < minimum:
            required = ".".join(str(item) for item in minimum)
            raise VerificationError(
                f"{fields[0]} requires Driver {required} or newer; found {fields[2]}"
            )
        values.append(
            {"name": fields[0], "compute_capability": fields[1], "driver": fields[2]}
        )
    if not values:
        raise VerificationError("No NVIDIA GPU was exposed to the container")
    return values


def verify(args: argparse.Namespace) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise VerificationError("The image supports only Linux x86_64")
    if version(platform.python_version())[:2] != (3, 11):
        raise VerificationError(f"Python 3.11 is required; found {platform.python_version()}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise VerificationError(f"Unable to import ONNX Runtime: {exc}") from exc
    if ort.__version__ != "1.27.0":
        raise VerificationError(f"ONNX Runtime 1.27.0 is required; found {ort.__version__}")
    providers = list(ort.get_available_providers())
    if args.provider not in providers:
        raise VerificationError(f"{args.provider} is unavailable; providers={providers}")
    if os.getenv("INSIGHTFACE_EXECUTION_PROVIDER") != args.provider:
        raise VerificationError("Configured Execution Provider does not match the image")
    if os.getenv("INSIGHTFACE_INFERENCE_MODE") != "onnx":
        raise VerificationError("Published images require ONNX inference mode")

    output: dict[str, Any] = {
        "mode": args.mode,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "onnx_runtime": ort.__version__,
        "available_execution_providers": providers,
        "required_execution_provider": args.provider,
    }
    if args.provider == "CUDAExecutionProvider":
        cuda = cuda_version()
        cudnn = cudnn_version()
        package = package_version("libcudnn9-cuda-12")
        if version(cuda)[:2] != (12, 9) or os.getenv("CUDA_VERSION") != "12.9.1":
            raise VerificationError(
                f"CUDA Runtime/Image 12.9.1 is required; found {cuda}/{os.getenv('CUDA_VERSION')}"
            )
        if cudnn != "9.24.0" or package != "9.24.0.43-1":
            raise VerificationError(
                f"cuDNN 9.24.0 package 9.24.0.43-1 is required; found {cudnn}/{package}"
            )
        if os.getenv("INSIGHTFACE_STRICT_CUDA") != "1":
            raise VerificationError("The CUDA image requires strict Provider auditing")
        output.update(
            {
                "cuda_runtime": cuda,
                "cuda_image": os.getenv("CUDA_VERSION"),
                "cudnn": cudnn,
                "cudnn_package": package,
                "gpus": gpu_details() if args.mode == "startup" else [],
                "strict_cuda": True,
                "model_session_validation": "application_lifespan_before_readiness",
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("build", "startup"))
    parser.add_argument("--provider", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        details = verify(args)
    except VerificationError as exc:
        print(f"InsightFace runtime verification failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print("InsightFace runtime verification: " + json.dumps(details, sort_keys=True), flush=True)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if args.mode == "startup" and not command:
        print("No server command supplied", file=sys.stderr)
        return 1
    if command:
        os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
