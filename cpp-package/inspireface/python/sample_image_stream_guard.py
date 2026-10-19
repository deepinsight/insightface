#!/usr/bin/env python3
"""Correctness, lifetime, validation, and latency gate for ImageStream."""

import ctypes
import gc
import statistics
import time
import weakref

import numpy as np

from inspireface.modules.core import (
    HF_CAMERA_ROTATION_0,
    HF_CAMERA_ROTATION_90,
    HF_STREAM_BGRA,
    HF_STREAM_BGR,
    HF_STREAM_GRAY,
    HF_STREAM_I420,
    HF_STREAM_YUV_NV21,
    HF_STREAM_RGB,
    HFCreateImageStream,
    HFCreateImageBitmapFromImageStreamProcess,
    HFDeBugGetUnreleasedStreamsCount,
    HFImageData,
    HFImageBitmap,
    HFImageBitmapData,
    HFImageBitmapGetData,
    HFImageStream,
    HFReleaseImageBitmap,
    HFReleaseImageStream,
)
from inspireface.modules.exception import InvalidInputError, check_error
from inspireface.modules.inspireface import ImageStream


BASELINE_CONTIGUOUS_P50_MS = 0.002834
BASELINE_CONTIGUOUS_P95_MS = 0.005542
CONTIGUOUS_ITERATIONS = 2000
NONCONTIGUOUS_ITERATIONS = 200


class Guard:
    def __init__(self):
        self.groups = {
            "pixel_accuracy": True,
            "input_ownership": True,
            "format_contract": True,
            "invalid_input": True,
            "lifecycle": True,
            "latency": True,
        }
        self.failures = []

    def expect(self, condition, group, message):
        if condition:
            return
        self.groups[group] = False
        self.failures.append(f"{group}: {message}")

    def expect_invalid(self, operation, message):
        try:
            operation()
        except InvalidInputError:
            return
        except Exception as error:
            self.expect(
                False,
                "invalid_input",
                f"{message}: raised {type(error).__name__} instead of InvalidInputError",
            )
            return
        self.expect(False, "invalid_input", f"{message}: input was accepted")

    def passed(self):
        return not self.failures


def stream_count():
    count = ctypes.c_int32()
    check_error(
        HFDeBugGetUnreleasedStreamsCount(ctypes.byref(count)),
        "Query unreleased ImageStream count",
    )
    return count.value


def decode_handle(handle):
    bitmap = HFImageBitmap()
    check_error(
        HFCreateImageBitmapFromImageStreamProcess(
            handle,
            ctypes.byref(bitmap),
            0,
            1.0,
        ),
        "Decode ImageStream into bitmap",
    )
    try:
        bitmap_data = HFImageBitmapData()
        check_error(
            HFImageBitmapGetData(bitmap, ctypes.byref(bitmap_data)),
            "Read decoded bitmap",
        )
        element_count = (
            bitmap_data.width * bitmap_data.height * bitmap_data.channels
        )
        flat = np.ctypeslib.as_array(
            bitmap_data.data,
            shape=(element_count,),
        )
        return flat.copy().reshape(
            bitmap_data.height,
            bitmap_data.width,
            bitmap_data.channels,
        )
    finally:
        check_error(HFReleaseImageBitmap(bitmap), "Release decoded bitmap")


def decode_stream(stream):
    return decode_handle(stream.handle)


def decode_legacy_ndarray(
    image,
    stream_format=HF_STREAM_BGR,
    width=None,
    height=None,
):
    """Decode through the pre-change pointer-only construction contract."""
    if not image.flags.c_contiguous:
        raise ValueError("legacy reference input must be contiguous")
    image_data = HFImageData()
    image_data.data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    image_data.width = image.shape[1] if width is None else width
    image_data.height = image.shape[0] if height is None else height
    image_data.format = stream_format
    image_data.rotation = HF_CAMERA_ROTATION_0
    handle = HFImageStream()
    check_error(
        HFCreateImageStream(ctypes.byref(image_data), ctypes.byref(handle)),
        "Create legacy reference ImageStream",
    )
    try:
        return decode_handle(handle)
    finally:
        check_error(HFReleaseImageStream(handle), "Release legacy reference ImageStream")


def max_pixel_delta(left, right):
    return int(np.abs(left.astype(np.int16) - right.astype(np.int16)).max())


def make_pattern(height, width, seed):
    y, x = np.indices((height, width), dtype=np.uint32)
    channels = [
        (x * 17 + y * 3 + seed * 11) % 256,
        (x * 5 + y * 29 + seed * 7) % 256,
        ((x ^ y) * 13 + seed * 19) % 256,
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


def benchmark(image, iterations):
    for _ in range(100):
        stream = ImageStream.load_from_cv_image(image)
        stream.release()

    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        stream = ImageStream.load_from_cv_image(image)
        stream.release()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "p50": statistics.median(samples),
        "p95": percentile(samples, 0.95),
        "mean": statistics.mean(samples),
    }


def run_pixel_accuracy_cases(guard):
    cases = [
        (7, 11, 1),
        (29, 37, 2),
        (64, 96, 3),
        (91, 127, 4),
        (160, 160, 5),
    ]
    for height, width, seed in cases:
        image = make_pattern(height, width, seed)
        legacy_decoded = decode_legacy_ndarray(image)
        with ImageStream.load_from_cv_image(image) as stream:
            decoded = decode_stream(stream)
            guard.expect(
                np.array_equal(decoded, legacy_decoded),
                "pixel_accuracy",
                f"BGR pattern {height}x{width} differs from legacy construction",
            )
            guard.expect(
                max_pixel_delta(decoded, image) <= 1,
                "pixel_accuracy",
                f"BGR pattern {height}x{width} exceeds native interpolation tolerance",
            )
            guard.expect(
                stream._data_owner is image,
                "input_ownership",
                f"contiguous BGR pattern {height}x{width} was copied",
            )

    source = make_pattern(73, 202, 17)
    roi = source[:, ::2, :]
    expected = roi.copy()
    legacy_decoded = decode_legacy_ndarray(expected)
    guard.expect(not roi.flags.c_contiguous, "pixel_accuracy", "ROI fixture is contiguous")
    with ImageStream.load_from_cv_image(roi) as stream:
        decoded = decode_stream(stream)
        guard.expect(
            np.array_equal(decoded, legacy_decoded),
            "pixel_accuracy",
            "non-contiguous ROI differs from its contiguous legacy reference",
        )
        guard.expect(
            stream._data_owner is not roi and stream._data_owner.flags.c_contiguous,
            "input_ownership",
            "non-contiguous ROI was not retained as a contiguous owner",
        )


def run_format_and_buffer_cases(guard):
    bgr = make_pattern(31, 47, 23)
    alpha = np.full(bgr.shape[:2] + (1,), 173, dtype=np.uint8)
    bgra = np.concatenate((bgr, alpha), axis=2)
    legacy_bgra = decode_legacy_ndarray(bgra, HF_STREAM_BGRA)
    with ImageStream.load_from_cv_image(bgra) as stream:
        guard.expect(
            stream.data_format == HF_STREAM_BGRA,
            "format_contract",
            "four-channel OpenCV image did not infer BGRA",
        )
        guard.expect(
            np.array_equal(decode_stream(stream), legacy_bgra),
            "pixel_accuracy",
            "BGRA default inference differs from explicit legacy BGRA decoding",
        )

    rgb = bgr[:, :, ::-1].copy()
    legacy_rgb = decode_legacy_ndarray(rgb, HF_STREAM_RGB)
    with ImageStream.load_from_cv_image(rgb, HF_STREAM_RGB) as stream:
        guard.expect(
            np.array_equal(decode_stream(stream), legacy_rgb),
            "pixel_accuracy",
            "explicit RGB input differs from legacy RGB decoding",
        )

    payload = bgr.tobytes()
    legacy_bgr = decode_legacy_ndarray(bgr)
    with ImageStream.load_from_buffer(
        payload,
        bgr.shape[1],
        bgr.shape[0],
        HF_STREAM_BGR,
        HF_CAMERA_ROTATION_0,
    ) as stream:
        guard.expect(
            stream._data_owner is payload,
            "input_ownership",
            "immutable bytes input was not retained",
        )
        guard.expect(
            np.array_equal(decode_stream(stream), legacy_bgr),
            "pixel_accuracy",
            "bytes input differs from legacy BGR decoding",
        )

    mutable_payload = bytearray(payload)
    with ImageStream.load_from_buffer(
        mutable_payload,
        bgr.shape[1],
        bgr.shape[0],
        HF_STREAM_BGR,
        HF_CAMERA_ROTATION_0,
    ) as stream:
        mutable_payload[0] ^= 0xFF
        expected = np.frombuffer(mutable_payload, dtype=np.uint8).reshape(bgr.shape).copy()
        legacy_mutated = decode_legacy_ndarray(expected)
        guard.expect(
            np.array_equal(decode_stream(stream), legacy_mutated),
            "input_ownership",
            "writable buffer was copied instead of retained zero-copy",
        )
    try:
        mutable_payload.extend(b"\x00")
    except BufferError:
        guard.expect(
            False,
            "input_ownership",
            "release kept a writable buffer export alive",
        )

    expanded = bytearray(len(payload) * 2)
    expanded[::2] = payload
    strided_view = memoryview(expanded)[::2]
    with ImageStream.load_from_buffer(
        strided_view,
        bgr.shape[1],
        bgr.shape[0],
        HF_STREAM_BGR,
        HF_CAMERA_ROTATION_0,
    ) as stream:
        guard.expect(
            np.array_equal(decode_stream(stream), legacy_bgr),
            "pixel_accuracy",
            "strided buffer was not copied in logical byte order",
        )

    ctypes_buffer = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    ctypes_pointer = ctypes.cast(ctypes_buffer, ctypes.POINTER(ctypes.c_uint8))
    with ImageStream.load_from_buffer(
        ctypes_pointer,
        bgr.shape[1],
        bgr.shape[0],
        HF_STREAM_BGR,
        HF_CAMERA_ROTATION_0,
    ) as stream:
        guard.expect(
            stream._data_owner is ctypes_pointer,
            "input_ownership",
            "ctypes pointer object was not retained",
        )
        guard.expect(
            np.array_equal(decode_stream(stream), legacy_bgr),
            "pixel_accuracy",
            "ctypes pointer was interpreted as pointer-value bytes",
        )

    yuv_width = 10
    yuv_height = 9
    yuv = np.arange(
        yuv_width * (yuv_height * 3 // 2),
        dtype=np.uint8,
    ).reshape(yuv_height * 3 // 2, yuv_width)
    legacy_yuv = decode_legacy_ndarray(
        yuv,
        HF_STREAM_YUV_NV21,
        width=yuv_width,
        height=yuv_height,
    )
    with ImageStream.load_from_ndarray(
        yuv,
        yuv_width,
        yuv_height,
        HF_STREAM_YUV_NV21,
        HF_CAMERA_ROTATION_0,
    ) as stream:
        guard.expect(
            np.array_equal(decode_stream(stream), legacy_yuv),
            "pixel_accuracy",
            "odd-height NV21 input differs from legacy NV21 decoding",
        )


def run_lifetime_cases(guard):
    image = make_pattern(53, 79, 31)
    expected = image.copy()
    legacy_decoded = decode_legacy_ndarray(expected)
    image_ref = weakref.ref(image)
    stream = ImageStream.load_from_cv_image(image)
    del image
    gc.collect()
    guard.expect(
        image_ref() is not None,
        "input_ownership",
        "ImageStream did not retain its source ndarray",
    )
    guard.expect(
        np.array_equal(decode_stream(stream), legacy_decoded),
        "pixel_accuracy",
        "source ndarray lifetime differs from legacy decoding",
    )
    stream.release()
    stream.release()
    guard.expect(stream.handle is None, "lifecycle", "release was not idempotent")
    guard.expect(
        stream._data_owner is None,
        "input_ownership",
        "release did not drop the retained input owner",
    )
    gc.collect()
    guard.expect(
        image_ref() is None,
        "input_ownership",
        "release kept the source ndarray alive",
    )

    context_stream = None
    with ImageStream.load_from_cv_image(expected) as context_stream:
        guard.expect(context_stream.handle is not None, "lifecycle", "context stream was closed early")
    guard.expect(context_stream.handle is None, "lifecycle", "context manager did not release stream")


def run_invalid_input_cases(guard):
    valid = make_pattern(12, 18, 41)
    guard.expect_invalid(
        lambda: ImageStream.load_from_cv_image(valid.astype(np.float32)),
        "float32 OpenCV image",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_cv_image(np.empty((0, 18, 3), dtype=np.uint8)),
        "zero-height OpenCV image",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_cv_image(valid, HF_STREAM_BGRA),
        "channel and format mismatch",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_buffer(
            valid.tobytes()[:-1],
            valid.shape[1],
            valid.shape[0],
            HF_STREAM_BGR,
            HF_CAMERA_ROTATION_0,
        ),
        "short byte buffer",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_ndarray(
            valid,
            valid.shape[1] + 1,
            valid.shape[0],
            HF_STREAM_BGR,
            HF_CAMERA_ROTATION_0,
        ),
        "ndarray shape mismatch",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_buffer(
            bytes(10 * (9 * 3 // 2) - 1),
            10,
            9,
            HF_STREAM_I420,
            HF_CAMERA_ROTATION_0,
        ),
        "short YUV420 buffer",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_buffer(
            valid.tobytes(),
            valid.shape[1],
            valid.shape[0],
            999,
            HF_CAMERA_ROTATION_0,
        ),
        "unknown stream format",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_buffer(
            valid.tobytes(),
            valid.shape[1],
            valid.shape[0],
            HF_STREAM_BGR,
            999,
        ),
        "unknown rotation",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_buffer(
            object(),
            valid.shape[1],
            valid.shape[0],
            HF_STREAM_BGR,
            HF_CAMERA_ROTATION_90,
        ),
        "object without a buffer",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_buffer(
            ctypes.POINTER(ctypes.c_uint8)(),
            valid.shape[1],
            valid.shape[0],
            HF_STREAM_BGR,
            HF_CAMERA_ROTATION_0,
        ),
        "null ctypes pointer",
    )
    guard.expect_invalid(
        lambda: ImageStream.load_from_buffer(
            valid.tobytes(),
            True,
            valid.shape[0],
            HF_STREAM_GRAY,
            HF_CAMERA_ROTATION_0,
        ),
        "boolean width",
    )


def main():
    guard = Guard()
    initial_streams = stream_count()
    started = time.perf_counter()

    run_pixel_accuracy_cases(guard)
    run_format_and_buffer_cases(guard)
    run_lifetime_cases(guard)
    run_invalid_input_cases(guard)

    contiguous = np.zeros((480, 640, 3), dtype=np.uint8)
    contiguous_latency = benchmark(contiguous, CONTIGUOUS_ITERATIONS)
    wide = np.zeros((480, 1280, 3), dtype=np.uint8)
    noncontiguous_latency = benchmark(wide[:, ::2, :], NONCONTIGUOUS_ITERATIONS)

    p50_limit = max(0.020, BASELINE_CONTIGUOUS_P50_MS * 2.0)
    p95_limit = max(0.050, BASELINE_CONTIGUOUS_P95_MS * 3.0)
    guard.expect(
        contiguous_latency["p50"] <= p50_limit,
        "latency",
        f"contiguous p50 {contiguous_latency['p50']:.6f} ms exceeds {p50_limit:.6f} ms",
    )
    guard.expect(
        contiguous_latency["p95"] <= p95_limit,
        "latency",
        f"contiguous p95 {contiguous_latency['p95']:.6f} ms exceeds {p95_limit:.6f} ms",
    )
    guard.expect(
        noncontiguous_latency["p95"] <= 2.500,
        "latency",
        f"non-contiguous p95 {noncontiguous_latency['p95']:.6f} ms exceeds 2.500 ms",
    )

    gc.collect()
    final_streams = stream_count()
    guard.expect(
        final_streams == initial_streams,
        "lifecycle",
        f"unreleased stream count changed from {initial_streams} to {final_streams}",
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print("ImageStream guard results")
    print(f"  native stream count: {initial_streams} -> {final_streams}")
    print("  contiguous create/release latency (480x640 BGR):")
    print(
        "    before: "
        f"p50={BASELINE_CONTIGUOUS_P50_MS:.6f} ms, "
        f"p95={BASELINE_CONTIGUOUS_P95_MS:.6f} ms"
    )
    print(
        "    after:  "
        f"p50={contiguous_latency['p50']:.6f} ms, "
        f"p95={contiguous_latency['p95']:.6f} ms, "
        f"mean={contiguous_latency['mean']:.6f} ms"
    )
    print(
        "    ratio:  "
        f"p50={contiguous_latency['p50'] / BASELINE_CONTIGUOUS_P50_MS:.3f}x, "
        f"p95={contiguous_latency['p95'] / BASELINE_CONTIGUOUS_P95_MS:.3f}x"
    )
    print("  non-contiguous copy latency (480x640 BGR logical image):")
    print(
        "    after:  "
        f"p50={noncontiguous_latency['p50']:.6f} ms, "
        f"p95={noncontiguous_latency['p95']:.6f} ms, "
        f"mean={noncontiguous_latency['mean']:.6f} ms"
    )
    for group, passed in guard.groups.items():
        print(f"  {group}: {'PASS' if passed else 'FAIL'}")
    print(f"  total: {elapsed_ms:.3f} ms")

    if guard.failures:
        print("Failures:")
        for failure in guard.failures:
            print(f"  - {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
