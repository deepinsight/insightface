"""Common assertions and native-resource helpers used by the sample suite."""

import ctypes
import gc
import threading
from contextlib import contextmanager

import cv2
import numpy as np

import inspireface as ifac
from inspireface.modules.core import (
    HFCreateImageBitmapFromImageStreamProcess,
    HFDeBugGetUnreleasedSessionsCount,
    HFDeBugGetUnreleasedStreamsCount,
    HFImageBitmap,
    HFImageBitmapData,
    HFImageBitmapGetData,
    HFReleaseImageBitmap,
)
from inspireface.modules.exception import check_error
from inspireface.param import HF_DETECT_MODE_ALWAYS_DETECT

from .settings import data_path


_METRICS = []
_METRICS_LOCK = threading.Lock()


def record_metric(name, value, unit="ms", **context):
    item = {"name": name, "value": float(value), "unit": unit}
    item.update(context)
    with _METRICS_LOCK:
        _METRICS.append(item)


def collected_metrics():
    with _METRICS_LOCK:
        return list(_METRICS)


def load_image(relative_path):
    path = data_path(relative_path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Unable to decode shared test image: {}".format(path))
    return image


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def image_mse(left, right):
    if left.shape != right.shape:
        return float("inf")
    difference = left.astype(np.float64) / 255.0 - right.astype(np.float64) / 255.0
    return float(np.mean(difference * difference))


def native_resource_count(counter, operation):
    count = ctypes.c_int32()
    check_error(counter(ctypes.byref(count)), operation)
    return count.value


def unreleased_session_count():
    return native_resource_count(
        HFDeBugGetUnreleasedSessionsCount,
        "Query unreleased session count",
    )


def unreleased_stream_count():
    return native_resource_count(
        HFDeBugGetUnreleasedStreamsCount,
        "Query unreleased stream count",
    )


@contextmanager
def managed_session(
    parameter,
    detect_mode=HF_DETECT_MODE_ALWAYS_DETECT,
    max_detect_num=10,
    detect_pixel_level=-1,
    track_by_detect_mode_fps=-1,
):
    session = ifac.InspireFaceSession(
        parameter,
        detect_mode,
        max_detect_num,
        detect_pixel_level,
        track_by_detect_mode_fps,
    )
    try:
        yield session
    finally:
        if session._sess is not None:
            session.release()
            session._sess = None


@contextmanager
def managed_feature_hub(configuration):
    ifac.feature_hub_enable(configuration)
    try:
        yield
    finally:
        ifac.feature_hub_disable()


def detect_from_fixture(session, relative_path):
    image = load_image(relative_path)
    return image, session.face_detection(image)


def extract_first_feature(session, relative_path):
    image, faces = detect_from_fixture(session, relative_path)
    if not faces:
        raise AssertionError("No face detected in {}".format(relative_path))
    return session.face_feature_extract(image, faces[0])


def decode_stream(stream, apply_rotation=True):
    bitmap = HFImageBitmap()
    check_error(
        HFCreateImageBitmapFromImageStreamProcess(
            stream.handle,
            ctypes.byref(bitmap),
            int(apply_rotation),
            1.0,
        ),
        "Decode ImageStream",
    )
    try:
        bitmap_data = HFImageBitmapData()
        check_error(
            HFImageBitmapGetData(bitmap, ctypes.byref(bitmap_data)),
            "Get decoded bitmap data",
        )
        size = bitmap_data.width * bitmap_data.height * bitmap_data.channels
        array = np.ctypeslib.as_array(bitmap_data.data, shape=(size,)).copy()
        return array.reshape(
            bitmap_data.height,
            bitmap_data.width,
            bitmap_data.channels,
        )
    finally:
        check_error(HFReleaseImageBitmap(bitmap), "Release decoded bitmap")


class NativeResourceCaseMixin(object):
    """Ensure each testcase returns session and stream counts to its baseline."""

    def setUp(self):
        super_method = getattr(super(), "setUp", None)
        if super_method is not None:
            super_method()
        gc.collect()
        self._session_count_before = unreleased_session_count()
        self._stream_count_before = unreleased_stream_count()

    def tearDown(self):
        gc.collect()
        self.assertEqual(
            unreleased_session_count(),
            self._session_count_before,
            "native session leak",
        )
        self.assertEqual(
            unreleased_stream_count(),
            self._stream_count_before,
            "native stream leak",
        )
        super_method = getattr(super(), "tearDown", None)
        if super_method is not None:
            super_method()
