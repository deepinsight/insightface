#!/usr/bin/env python3
"""End-to-end detection consistency and latency gate for ImageStream layouts."""

import argparse
import ctypes
import gc
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import inspireface as ifac
from inspireface.modules.core import (
    HFDeBugGetUnreleasedSessionsCount,
    HFDeBugGetUnreleasedStreamsCount,
)
from inspireface.modules.exception import check_error
from inspireface.param import HF_DETECT_MODE_ALWAYS_DETECT, HF_ENABLE_NONE


DEFAULT_IMAGE_NAMES = (
    "kun.jpg",
    "yifei.jpg",
    "face_sample.png",
    "view.jpg",
)
MAX_NONCONTIGUOUS_OVERHEAD_MS = 12.0


def native_count(function, operation):
    count = ctypes.c_int32()
    check_error(function(ctypes.byref(count)), operation)
    return count.value


def face_signature(faces):
    return [
        (
            face.location,
            round(face.roll, 6),
            round(face.yaw, 6),
            round(face.pitch, 6),
            round(face.detection_confidence, 6),
        )
        for face in faces
    ]


def detect_once(image):
    session = ifac.InspireFaceSession(
        HF_ENABLE_NONE,
        HF_DETECT_MODE_ALWAYS_DETECT,
    )
    try:
        started = time.perf_counter_ns()
        faces = session.face_detection(image)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return faces, elapsed_ms
    finally:
        session.release()


def make_noncontiguous_view(image):
    padded = np.zeros(
        (image.shape[0], image.shape[1] * 2, image.shape[2]),
        dtype=image.dtype,
    )
    padded[:, ::2, :] = image
    view = padded[:, ::2, :]
    if view.flags.c_contiguous:
        raise RuntimeError("non-contiguous fixture unexpectedly became contiguous")
    return padded, view


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "test_res" / "pack" / "Pikachu",
        help="Path to an InspireFace model pack",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_root / "test_res" / "data" / "bulk",
        help="Directory containing the guard images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model pack not found: {args.model}")

    initial_sessions = native_count(
        HFDeBugGetUnreleasedSessionsCount,
        "Query unreleased session count",
    )
    initial_streams = native_count(
        HFDeBugGetUnreleasedStreamsCount,
        "Query unreleased stream count",
    )
    ifac.launch(resource_path=str(args.model.resolve()))

    failures = []
    contiguous_latencies = []
    noncontiguous_latencies = []
    for image_name in DEFAULT_IMAGE_NAMES:
        image_path = args.image_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            failures.append(f"unable to read {image_path}")
            continue

        padded, view = make_noncontiguous_view(image)
        contiguous_faces, contiguous_ms = detect_once(image)
        noncontiguous_faces, noncontiguous_ms = detect_once(view)
        contiguous_signature = face_signature(contiguous_faces)
        noncontiguous_signature = face_signature(noncontiguous_faces)
        same = contiguous_signature == noncontiguous_signature
        if not same:
            failures.append(f"{image_name} detection result changed")

        contiguous_latencies.append(contiguous_ms)
        noncontiguous_latencies.append(noncontiguous_ms)
        print(
            f"{image_name}: faces={len(contiguous_faces)}, same={same}, "
            f"contiguous={contiguous_ms:.3f} ms, "
            f"noncontiguous={noncontiguous_ms:.3f} ms"
        )
        del padded, view

    overheads = [
        noncontiguous - contiguous
        for contiguous, noncontiguous in zip(
            contiguous_latencies,
            noncontiguous_latencies,
        )
    ]
    if overheads and max(overheads) > MAX_NONCONTIGUOUS_OVERHEAD_MS:
        failures.append(
            f"maximum non-contiguous overhead {max(overheads):.3f} ms exceeds "
            f"{MAX_NONCONTIGUOUS_OVERHEAD_MS:.3f} ms"
        )

    gc.collect()
    final_sessions = native_count(
        HFDeBugGetUnreleasedSessionsCount,
        "Query final unreleased session count",
    )
    final_streams = native_count(
        HFDeBugGetUnreleasedStreamsCount,
        "Query final unreleased stream count",
    )
    if final_sessions != initial_sessions:
        failures.append(
            f"unreleased session count changed {initial_sessions} -> {final_sessions}"
        )
    if final_streams != initial_streams:
        failures.append(
            f"unreleased stream count changed {initial_streams} -> {final_streams}"
        )

    if contiguous_latencies:
        print(
            "latency median: "
            f"contiguous={statistics.median(contiguous_latencies):.3f} ms, "
            f"noncontiguous={statistics.median(noncontiguous_latencies):.3f} ms, "
            f"copy overhead={statistics.median(overheads):.3f} ms"
        )
    print(f"native sessions: {initial_sessions} -> {final_sessions}")
    print(f"native streams: {initial_streams} -> {final_streams}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
