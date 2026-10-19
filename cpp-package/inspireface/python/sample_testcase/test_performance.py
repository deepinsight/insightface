"""Latency and optional extended benchmark cases."""

import statistics
import time
import unittest

import inspireface as ifac
from inspireface.param import HF_ENABLE_NONE

from .common import (
    NativeResourceCaseMixin,
    load_image,
    managed_session,
    record_metric,
)
from .settings import RUN_BENCHMARKS


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


class PerformanceCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_face_detection_latency_gate(self):
        image = load_image("bulk/kun.jpg")
        with managed_session(HF_ENABLE_NONE, detect_pixel_level=320) as session:
            for _ in range(3):
                self.assertEqual(len(session.face_detection(image)), 1)
            samples = []
            for _ in range(20):
                started = time.perf_counter_ns()
                faces = session.face_detection(image)
                samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
                self.assertEqual(len(faces), 1)
        median = statistics.median(samples)
        p95 = percentile(samples, 0.95)
        record_metric("face_detection_320_p50", median)
        record_metric("face_detection_320_p95", p95)
        self.assertLess(median, 50.0)
        self.assertLess(p95, 100.0)

    def test_image_stream_creation_latency_gate(self):
        image = load_image("bulk/kun.jpg")
        samples = []
        for _ in range(1000):
            started = time.perf_counter_ns()
            stream = ifac.ImageStream.load_from_cv_image(image)
            stream.release()
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        median = statistics.median(samples)
        p95 = percentile(samples, 0.95)
        record_metric("image_stream_create_release_p50", median)
        record_metric("image_stream_create_release_p95", p95)
        self.assertLess(median, 0.05)
        self.assertLess(p95, 0.10)

    @unittest.skipUnless(RUN_BENCHMARKS, "extended benchmarks require --benchmark")
    def test_detection_levels_extended_benchmark(self):
        image = load_image("bulk/pedestrian.png")
        for level in (192, 320, 640):
            with managed_session(
                HF_ENABLE_NONE,
                max_detect_num=25,
                detect_pixel_level=level,
            ) as session:
                session.set_track_preview_size(level)
                session.set_filter_minimum_face_pixel_size(0)
                samples = []
                for _ in range(30):
                    started = time.perf_counter_ns()
                    faces = session.face_detection(image)
                    samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
                    self.assertGreater(len(faces), 0)
            record_metric(
                "face_detection_level_p50",
                statistics.median(samples),
                level=level,
            )
            record_metric(
                "face_detection_level_p95",
                percentile(samples, 0.95),
                level=level,
            )
