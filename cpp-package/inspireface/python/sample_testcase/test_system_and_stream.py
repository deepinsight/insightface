"""System, native-resource, and ImageStream cases mirroring C++ API tests."""

import ctypes
import gc
import os
import subprocess
import sys
import time
import unittest
import weakref

import numpy as np

import inspireface as ifac
from inspireface.modules.core import (
    HF_LOG_ERROR,
    HFExecuteFaceTrackSnapshot,
    HFFaceResultSnapshot,
    HFGetFaceResultSnapshotData,
    HFMultipleFaceData,
    HFReleaseFaceResultSnapshot,
)
from inspireface.modules.exception import InvalidInputError, ResourceError, check_error
from inspireface.param import (
    HF_CAMERA_ROTATION_0,
    HF_DETECT_MODE_ALWAYS_DETECT,
    HF_ENABLE_NONE,
    HF_STREAM_BGR,
    HF_STREAM_I420,
)

from .common import (
    NativeResourceCaseMixin,
    decode_stream,
    load_image,
    managed_session,
    unreleased_session_count,
    unreleased_stream_count,
)
from .settings import MODEL_PATH


class SystemCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_resource_pack_validation_is_side_effect_free_and_bounded(self):
        launch_status = ifac.query_launch_status()
        started = time.perf_counter()
        info = ifac.validate_resource_pack(MODEL_PATH)
        elapsed = time.perf_counter() - started

        self.assertIsInstance(info, ifac.ResourcePackInfo)
        self.assertEqual(info.tag, "Pikachu")
        self.assertEqual(info.version, "4.0")
        self.assertTrue(info.major)
        self.assertTrue(info.release_date)
        self.assertGreater(info.archive_file_count, 1)
        self.assertGreater(info.model_count, 0)
        self.assertEqual(ifac.query_launch_status(), launch_status)
        self.assertLess(elapsed, 2.0)

        with self.assertRaises(ifac.InspireFaceError):
            ifac.validate_resource_pack(str(MODEL_PATH) + ".missing")
        self.assertEqual(ifac.query_launch_status(), launch_status)

    def test_owned_detection_snapshot_survives_python_resource_release(self):
        snapshot = HFFaceResultSnapshot()
        expected_token = None
        try:
            image = load_image("bulk/kun.jpg")
            with ifac.InspireFaceSession(HF_ENABLE_NONE) as session:
                with ifac.ImageStream.load_from_cv_image(image) as stream:
                    check_error(
                        HFExecuteFaceTrackSnapshot(
                            session._sess,
                            stream.handle,
                            ctypes.byref(snapshot),
                        ),
                        "Execute owned face tracking",
                    )
                    data = HFMultipleFaceData()
                    check_error(
                        HFGetFaceResultSnapshotData(snapshot, ctypes.byref(data)),
                        "Read owned face result",
                    )
                    self.assertGreater(data.detectedNum, 0)
                    self.assertGreater(data.tokens[0].size, 0)
                    expected_token = ctypes.string_at(
                        data.tokens[0].data,
                        data.tokens[0].size,
                    )

            persistent = HFMultipleFaceData()
            check_error(
                HFGetFaceResultSnapshotData(snapshot, ctypes.byref(persistent)),
                "Read persistent owned face result",
            )
            self.assertGreater(persistent.detectedNum, 0)
            self.assertEqual(
                ctypes.string_at(
                    persistent.tokens[0].data,
                    persistent.tokens[0].size,
                ),
                expected_token,
            )
        finally:
            if snapshot:
                check_error(
                    HFReleaseFaceResultSnapshot(snapshot),
                    "Release owned face result",
                )

    def test_launch_status_and_version(self):
        self.assertTrue(ifac.query_launch_status())
        self.assertEqual(ifac.c_api_level(), 2)
        parts = ifac.version().split(".")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(part.isdigit() for part in parts))

        expected_components = (
            "inspireface",
            "mnn",
            "inspirecv",
            "eigen",
            "sqlite",
            "sqlite_vec",
            "nlohmann_json",
            "opencv",
            "tensorrt",
            "cuda",
            "rknn",
            "rga",
            "coreml",
        )
        components = ifac.component_versions()
        self.assertEqual(tuple(components), expected_components)
        self.assertEqual(components, ifac.component_versions())
        for name in expected_components:
            component = components[name]
            self.assertIn(component["state"], {"known", "unknown", "disabled"})
            if component["state"] == "known":
                self.assertEqual(
                    component["version"],
                    f'{component["major"]}.{component["minor"]}.{component["patch"]}',
                )
            else:
                self.assertIsNone(component["version"])
                self.assertIsNone(component["major"])
                self.assertIsNone(component["minor"])
                self.assertIsNone(component["patch"])
        for name in expected_components[:7]:
            self.assertEqual(components[name]["state"], "known")

        diagnostic = ifac.diagnostic_info()
        self.assertTrue(diagnostic.startswith("InspireFace SDK "))
        self.assertIn("\nComponents: inspireface=", diagnostic)
        rendered = dict(
            item.split("=", 1)
            for item in diagnostic.rsplit("\nComponents: ", 1)[1].split(";")
        )
        self.assertEqual(tuple(rendered), expected_components)
        for name, component in components.items():
            expected = component["version"] or component["state"]
            self.assertEqual(rendered[name], expected)

        started = time.perf_counter()
        for _ in range(200):
            self.assertEqual(ifac.diagnostic_info(), diagnostic)
        self.assertLess(time.perf_counter() - started, 2.0)

    def test_global_lifecycle_in_isolated_process(self):
        source = """
import os
import inspireface as ifac
model = os.environ['INSPIREFACE_SYSTEM_MODEL']
assert not ifac.query_launch_status()
components = ifac.component_versions()
assert components['inspireface']['state'] == 'known'
assert components['mnn']['state'] == 'known'
assert '\\nComponents: inspireface=' in ifac.diagnostic_info()
assert not ifac.query_launch_status()
assert ifac.launch(resource_path=model)
assert ifac.query_launch_status()
assert ifac.reload(resource_path=model)
assert ifac.query_launch_status()
assert ifac.terminate()
assert not ifac.query_launch_status()
"""
        environment = os.environ.copy()
        environment["INSPIREFACE_SYSTEM_MODEL"] = str(MODEL_PATH)
        completed = subprocess.run(
            [sys.executable, "-c", source],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_session_resource_registry(self):
        baseline = unreleased_session_count()
        sessions = [
            ifac.InspireFaceSession(
                HF_ENABLE_NONE,
                HF_DETECT_MODE_ALWAYS_DETECT,
            )
            for _ in range(6)
        ]
        self.assertEqual(unreleased_session_count(), baseline + len(sessions))
        for session in sessions[::2]:
            session.release()
            session._sess = None
        self.assertEqual(unreleased_session_count(), baseline + 3)
        for session in sessions[1::2]:
            session.release()
            session._sess = None
        self.assertEqual(unreleased_session_count(), baseline)

    def test_session_context_and_temporary_streams_release_deterministically(self):
        baseline_sessions = unreleased_session_count()
        image = load_image("bulk/kun.jpg")
        with ifac.InspireFaceSession(HF_ENABLE_NONE) as session:
            baseline_streams = unreleased_stream_count()
            for _ in range(4):
                self.assertGreater(len(session.face_detection(image)), 0)
                self.assertEqual(unreleased_stream_count(), baseline_streams)
        self.assertEqual(unreleased_session_count(), baseline_sessions)

    def test_stream_resource_registry(self):
        image = load_image("bulk/pedestrian.png")
        baseline = unreleased_stream_count()
        streams = [ifac.ImageStream.load_from_cv_image(image) for _ in range(8)]
        self.assertEqual(unreleased_stream_count(), baseline + len(streams))
        for stream in streams[:3]:
            stream.release()
        self.assertEqual(unreleased_stream_count(), baseline + 5)
        for stream in streams[3:]:
            stream.release()
        self.assertEqual(unreleased_stream_count(), baseline)

    def test_session_runtime_setters(self):
        with managed_session(HF_ENABLE_NONE) as session:
            session.set_detection_confidence_threshold(0.5)
            session.set_track_preview_size(320)
            session.set_filter_minimum_face_pixel_size(16)
            session.set_track_mode_smooth_ratio(0.025)
            session.set_track_mode_num_smooth_cache_frame(5)
            session.set_track_model_detect_interval(2)
            session.set_landmark_augmentation_num(1)
            session.set_landmark_augmentation_num(3)
            session.set_track_lost_recovery_mode(False)
            session.set_enable_track_cost_spend(False)

    def test_landmark_augmentation_setter_rejects_invalid_values(self):
        with managed_session(HF_ENABLE_NONE) as session:
            with self.assertRaises(InvalidInputError):
                session.set_landmark_augmentation_num(0)

    def test_logging_controls(self):
        ifac.set_logging_level(HF_LOG_ERROR)
        ifac.disable_logging()
        ifac.set_logging_level(HF_LOG_ERROR)


class ImageStreamCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_shared_images_match_contiguous_references(self):
        for relative_path in (
            "bulk/kun.jpg",
            "bulk/yifei.jpg",
            "bulk/face_sample.png",
            "bulk/view.jpg",
        ):
            image = load_image(relative_path)
            padded = np.zeros(
                (image.shape[0], image.shape[1] * 2, image.shape[2]),
                dtype=image.dtype,
            )
            padded[:, ::2, :] = image
            view = padded[:, ::2, :]
            self.assertFalse(view.flags.c_contiguous)
            with ifac.ImageStream.load_from_cv_image(image) as reference_stream:
                reference = decode_stream(reference_stream, apply_rotation=False)
            with ifac.ImageStream.load_from_cv_image(view) as view_stream:
                actual = decode_stream(view_stream, apply_rotation=False)
                self.assertTrue(view_stream._data_owner.flags.c_contiguous)
            self.assertTrue(np.array_equal(actual, reference), relative_path)

    def test_cv_bgra_inference_and_buffer_inputs(self):
        bgr = load_image("bulk/kun_crop.jpg")
        alpha = np.full(bgr.shape[:2] + (1,), 127, dtype=np.uint8)
        bgra = np.concatenate((bgr, alpha), axis=2)
        with ifac.ImageStream.load_from_cv_image(bgra) as stream:
            self.assertEqual(stream.data_format, ifac.HF_STREAM_BGRA)
            decoded_bgra = decode_stream(stream, apply_rotation=False)
        with ifac.ImageStream.load_from_cv_image(bgr) as stream:
            decoded_bgr = decode_stream(stream, apply_rotation=False)
        self.assertLessEqual(
            int(
                np.max(
                    np.abs(
                        decoded_bgra.astype(np.int16)
                        - decoded_bgr.astype(np.int16)
                    )
                )
            ),
            1,
        )

        payload = bytearray(bgr.tobytes())
        with ifac.ImageStream.load_from_buffer(
            payload,
            bgr.shape[1],
            bgr.shape[0],
            HF_STREAM_BGR,
            HF_CAMERA_ROTATION_0,
        ) as stream:
            payload[0] ^= 0xFF
            decoded = decode_stream(stream, apply_rotation=False)
        expected = np.frombuffer(payload, dtype=np.uint8).reshape(bgr.shape).copy()
        self.assertLessEqual(
            int(np.max(np.abs(decoded.astype(np.int16) - expected.astype(np.int16)))),
            1,
        )
        payload.extend(b"\x00")

    def test_gray_image_and_yuv_dimension_contract(self):
        bgr = load_image("bulk/kun_crop.jpg")
        gray = np.ascontiguousarray(bgr[:, :, 0])
        with ifac.ImageStream.load_from_cv_image(gray) as stream:
            self.assertEqual(stream.data_format, ifac.HF_STREAM_GRAY)
            decoded = decode_stream(stream, apply_rotation=False)
        self.assertEqual(decoded.shape, gray.shape + (3,))
        np.testing.assert_array_equal(decoded[:, :, 0], decoded[:, :, 1])
        np.testing.assert_array_equal(decoded[:, :, 1], decoded[:, :, 2])
        difference = decoded[:, :, 0].astype(np.int16) - gray.astype(np.int16)
        self.assertLess(float(np.mean((difference / 255.0) ** 2)), 2e-5)
        self.assertLessEqual(int(np.max(np.abs(difference))), 64)

        with self.assertRaises(InvalidInputError):
            ifac.ImageStream.load_from_ndarray(
                np.zeros((4, 3), dtype=np.uint8),
                3,
                3,
                HF_STREAM_I420,
                HF_CAMERA_ROTATION_0,
            )

    def test_query_dma_heap_path_uses_owned_output_buffer(self):
        path = ifac.query_expansive_hardware_rockchip_dma_heap_path()
        self.assertIsInstance(path, str)

    def test_source_lifetime_and_context_release(self):
        image = load_image("bulk/yifei.jpg")
        with ifac.ImageStream.load_from_cv_image(image) as reference_stream:
            expected = decode_stream(reference_stream, apply_rotation=False)
        image_reference = weakref.ref(image)
        stream = ifac.ImageStream.load_from_cv_image(image)
        del image
        gc.collect()
        self.assertIsNotNone(image_reference())
        decoded = decode_stream(stream, apply_rotation=False)
        self.assertLessEqual(
            int(np.max(np.abs(decoded.astype(np.int16) - expected.astype(np.int16)))),
            1,
        )
        stream.release()
        stream.release()
        gc.collect()
        self.assertIsNone(image_reference())
        self.assertIsNone(stream.handle)
        with self.assertRaises(ResourceError):
            stream.write_to_file("released-stream.jpg")

    def test_invalid_stream_contract(self):
        image = load_image("bulk/kun_crop.jpg")
        invalid_operations = (
            lambda: ifac.ImageStream.load_from_cv_image(image.astype(np.float32)),
            lambda: ifac.ImageStream.load_from_cv_image(image, ifac.HF_STREAM_BGRA),
            lambda: ifac.ImageStream.load_from_buffer(
                image.tobytes()[:-1],
                image.shape[1],
                image.shape[0],
                HF_STREAM_BGR,
                HF_CAMERA_ROTATION_0,
            ),
            lambda: ifac.ImageStream.load_from_buffer(
                ctypes.POINTER(ctypes.c_uint8)(),
                image.shape[1],
                image.shape[0],
                HF_STREAM_BGR,
                HF_CAMERA_ROTATION_0,
            ),
            lambda: ifac.ImageStream.load_from_buffer(
                image.tobytes(),
                0,
                image.shape[0],
                HF_STREAM_BGR,
                HF_CAMERA_ROTATION_0,
            ),
        )
        for operation in invalid_operations:
            with self.assertRaises(InvalidInputError):
                operation()
