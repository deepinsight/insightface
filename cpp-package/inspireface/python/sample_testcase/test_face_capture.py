"""Face-capture binding contract, accuracy, lifecycle, concurrency, and latency gates."""

import ctypes
import math
import threading
import time
import unittest

import inspireface as ifac
from inspireface.modules import herror as errcode
from inspireface.modules.core import (
    HFFaceCaptureConfig,
    HFFaceCaptureMetrics,
    HFFaceCaptureProgress,
    HFFaceCaptureResult,
)
from inspireface.modules.exception import InvalidInputError, ProcessingError, ResourceError
from inspireface.param import (
    HF_DETECT_MODE_LIGHT_TRACK,
    HF_ENABLE_NONE,
)

from .common import NativeResourceCaseMixin, calculate_iou, load_image, managed_session, record_metric


def immediate_config(filter_mask=ifac.FaceCaptureFilter.FACE_COUNT):
    return ifac.FaceCaptureConfig(
        filter_mask=int(filter_mask),
        output_count=1,
        stable_duration_ms=0,
        collect_duration_ms=0,
        max_collect_duration_ms=1000,
        min_candidate_interval_ms=0,
    )


class FaceCaptureCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_native_layout_defaults_and_python_constants_match(self):
        defaults = ifac.FaceCaptureConfig.defaults()
        native = defaults._native()
        self.assertEqual(native.structSize, ctypes.sizeof(HFFaceCaptureConfig))
        self.assertEqual(native.structVersion, 1)
        self.assertEqual(native.filterMask, defaults.filter_mask)
        self.assertEqual(native.outputCount, defaults.output_count)
        self.assertEqual(native.minTrackCount, 5)
        self.assertEqual(HFFaceCaptureConfig.minTrackCount.offset, 20)
        self.assertEqual(HFFaceCaptureProgress.trackCount.offset, 28)
        self.assertEqual(tuple(native.reserved), (0,) * 8)
        self.assertGreater(ctypes.sizeof(HFFaceCaptureConfig), ctypes.sizeof(HFFaceCaptureResult))
        self.assertGreater(ctypes.sizeof(HFFaceCaptureProgress), ctypes.sizeof(HFFaceCaptureMetrics))
        self.assertEqual(ifac.FaceCaptureFilter.BRIGHTNESS, 1 << 8)
        self.assertEqual(ifac.FaceCaptureFilter.TRACK_COUNT, 1 << 9)
        self.assertEqual(ifac.FaceCaptureRejectReason.BRIGHTNESS, 1 << 10)
        self.assertEqual(ifac.FaceCaptureRejectReason.TRACK_COUNT_TOO_LOW, 1 << 11)

    def test_invalid_config_and_required_features_raise_typed_errors(self):
        with managed_session(HF_ENABLE_NONE) as session:
            invalid_cases = (
                ifac.FaceCaptureConfig(filter_mask=1 << 63),
                ifac.FaceCaptureConfig(output_count=0),
                ifac.FaceCaptureConfig(min_track_count=0),
                ifac.FaceCaptureConfig(min_track_count=1 << 31),
                ifac.FaceCaptureConfig(min_face_width_ratio=float("nan")),
                ifac.FaceCaptureConfig(collect_duration_ms=1001, max_collect_duration_ms=1000),
            )
            for config in invalid_cases:
                with self.assertRaises(InvalidInputError) as raised:
                    session.create_face_capture(config)
                self.assertEqual(raised.exception.error_code, errcode.HERR_CAPTURE_INVALID_CONFIG)

            pose = immediate_config(ifac.FaceCaptureFilter.POSE)
            with self.assertRaises(ProcessingError) as raised:
                session.create_face_capture(pose)
            self.assertEqual(raised.exception.error_code, errcode.HERR_CAPTURE_REQUIRED_FEATURE_OFF)

        with self.assertRaises(InvalidInputError):
            ifac.FaceCaptureConfig(output_count=True)._native()
        with self.assertRaises(InvalidInputError):
            ifac.FaceCaptureConfig(stable_duration_ms=-1)._native()

    def test_track_count_gate_waits_for_fifth_frame_and_can_be_disabled(self):
        image = load_image("bulk/kun.jpg")
        config = immediate_config(ifac.FaceCaptureFilter.TRACK_COUNT)
        config.min_track_count = 5
        with managed_session(HF_ENABLE_NONE, detect_mode=HF_DETECT_MODE_LIGHT_TRACK) as session:
            with session.create_face_capture(config) as capture:
                for frame_id in range(1, 5):
                    progress = capture.update(image, frame_id, frame_id)
                    self.assertEqual(progress.track_count, frame_id)
                    self.assertTrue(
                        progress.reject_reasons & ifac.FaceCaptureRejectReason.TRACK_COUNT_TOO_LOW
                    )
                    self.assertEqual(capture.results(), [])
                progress = capture.update(image, 5, 5)
                self.assertEqual(progress.track_count, 5)
                self.assertEqual(progress.reject_reasons, ifac.FaceCaptureRejectReason.NONE)
                self.assertEqual(len(capture.results()), 1)

            disabled = immediate_config(ifac.FaceCaptureFilter.FACE_COUNT)
            disabled.min_track_count = (1 << 32) - 1
            with session.create_face_capture(disabled) as capture:
                progress = capture.update(image, 1, 1)
                self.assertEqual(progress.reject_reasons, ifac.FaceCaptureRejectReason.NONE)
                self.assertEqual(len(capture.results()), 1)

    def test_direct_update_selects_single_face_and_rejects_real_edge_images(self):
        config = immediate_config()
        with managed_session(HF_ENABLE_NONE, max_detect_num=20, detect_pixel_level=320) as session:
            with session.create_face_capture(config) as capture:
                single = load_image("bulk/kun.jpg")
                progress = capture.update(single, 1, 10)
                self.assertEqual(progress.state, ifac.FaceCaptureState.READY)
                self.assertEqual(progress.reject_reasons, ifac.FaceCaptureRejectReason.NONE)
                results = capture.results()
                self.assertEqual(len(results), 1)
                self.assertGreater(calculate_iou(results[0].face.location, (79, 104, 247, 271)), 0.5)
                self.assertEqual(session.get_face_five_key_points(results[0].face).shape, (5, 2))

                capture.reset()
                no_face = load_image("crop/no_face.png")
                progress = capture.update(no_face, 1, 10)
                self.assertEqual(progress.state, ifac.FaceCaptureState.IDLE)
                self.assertTrue(progress.reject_reasons & ifac.FaceCaptureRejectReason.NO_FACE)
                self.assertEqual(progress.progress, 0.0)
                self.assertEqual(capture.results(), [])

                capture.reset()
                multiple = load_image("bulk/pedestrian.png")
                progress = capture.update(multiple, 1, 10)
                self.assertTrue(progress.reject_reasons & ifac.FaceCaptureRejectReason.MULTIPLE_FACES)
                self.assertEqual(progress.candidate_count, 0)

    def test_snapshot_reuse_is_exact_and_capture_pins_released_parent_session(self):
        image = load_image("bulk/kun.jpg")
        session = ifac.InspireFaceSession(HF_ENABLE_NONE, max_detect_num=4)
        snapshot = session.face_detection_snapshot(image)
        capture = session.create_face_capture(immediate_config())
        try:
            self.assertEqual(len(snapshot.faces), 1)
            expected = snapshot.faces[0]
            session.release()
            progress = capture.update(image, 1, 1, snapshot=snapshot)
            self.assertEqual(progress.state, ifac.FaceCaptureState.READY)
            result = capture.results()[0]
            self.assertEqual(result.face.location, expected.location)
            self.assertEqual(result.face.track_id, expected.track_id)
            self.assertEqual(result.face.track_count, expected.track_count)
            self.assertAlmostEqual(result.face.roll, expected.roll, places=6)
            self.assertAlmostEqual(result.face.yaw, expected.yaw, places=6)
            self.assertAlmostEqual(result.face.pitch, expected.pitch, places=6)
        finally:
            capture.close()
            snapshot.close()
            session.close()
        self.assertTrue(capture.closed)
        self.assertTrue(snapshot.closed)
        capture.close()
        snapshot.close()

    def test_stability_order_finish_reset_and_copied_result_token_lifecycle(self):
        image = load_image("bulk/kun.jpg")
        config = immediate_config(ifac.FaceCaptureFilter.FACE_COUNT | ifac.FaceCaptureFilter.STABILITY)
        config.stable_duration_ms = 100
        config.max_collect_duration_ms = 1000
        with managed_session(
            HF_ENABLE_NONE,
            detect_mode=HF_DETECT_MODE_LIGHT_TRACK,
        ) as session:
            with session.face_detection_snapshot(image) as snapshot:
                with session.create_face_capture(config) as capture:
                    first = capture.update(image, 1, 10, snapshot)
                    middle = capture.update(image, 2, 60, snapshot)
                    ready = capture.update(image, 3, 110, snapshot)
                    self.assertEqual(first.state, ifac.FaceCaptureState.STABILIZING)
                    self.assertEqual(middle.state, ifac.FaceCaptureState.STABILIZING)
                    self.assertEqual(ready.state, ifac.FaceCaptureState.READY)
                    self.assertTrue(first.reject_reasons & ifac.FaceCaptureRejectReason.UNSTABLE)
                    self.assertEqual(ready.reject_reasons, ifac.FaceCaptureRejectReason.NONE)

                    with self.assertRaises(InvalidInputError) as raised:
                        capture.update(image, 3, 111, snapshot)
                    self.assertEqual(raised.exception.error_code, errcode.HERR_CAPTURE_FRAME_OUT_OF_ORDER)

                    copied_face = capture.results()[0].face
                    self.assertEqual(capture.finish().state, ifac.FaceCaptureState.FINISHED)
                    self.assertEqual(capture.finish().state, ifac.FaceCaptureState.FINISHED)
                    capture.reset()
                    self.assertEqual(capture.results(), [])
                    self.assertEqual(session.get_face_five_key_points(copied_face).shape, (5, 2))

        with self.assertRaises(ResourceError):
            capture.results()

    def test_independent_instances_are_deterministic_and_snapshot_path_is_bounded(self):
        image = load_image("bulk/kun.jpg")
        config = immediate_config(
            ifac.FaceCaptureFilter.FACE_COUNT
            | ifac.FaceCaptureFilter.FACE_SIZE
            | ifac.FaceCaptureFilter.FACE_POSITION
            | ifac.FaceCaptureFilter.TRACK_COUNT
        )
        config.min_track_count = 1
        config.output_count = 8
        config.max_collect_duration_ms = 1000000
        with managed_session(HF_ENABLE_NONE) as session:
            with session.face_detection_snapshot(image) as snapshot:
                with session.create_face_capture(config) as first, session.create_face_capture(config) as second:
                    failures = []

                    def run(capture):
                        try:
                            for index in range(1, 501):
                                capture.update(image, index, index, snapshot)
                        except Exception as error:  # pragma: no cover - asserted below
                            failures.append(error)

                    threads = (threading.Thread(target=run, args=(first,)), threading.Thread(target=run, args=(second,)))
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    self.assertEqual(failures, [])
                    first_results = first.results()
                    second_results = second.results()
                    self.assertEqual(len(first_results), 8)
                    self.assertEqual(
                        [(item.frame_id, item.score) for item in first_results],
                        [(item.frame_id, item.score) for item in second_results],
                    )

                with session.create_face_capture(config) as benchmark:
                    iterations = 2000
                    started = time.perf_counter()
                    for index in range(1, iterations + 1):
                        benchmark.update(image, index, index, snapshot)
                    average_us = (time.perf_counter() - started) * 1000000.0 / iterations
                    record_metric("python_face_capture_snapshot_update", average_us, unit="us")
                    self.assertLess(average_us, 1000.0)
                    self.assertLessEqual(len(benchmark.results()), 8)
                    self.assertTrue(math.isfinite(average_us))


if __name__ == "__main__":
    unittest.main()
