"""Detection, rotation, pose, landmarks, and preview-level regression cases."""

import ctypes
import unittest

import numpy as np

import inspireface as ifac
from inspireface.modules.core import (
    HFSessionGetTrackPreviewSize,
    HInt32,
)
from inspireface.modules.exception import check_error
from inspireface.param import (
    HF_CAMERA_ROTATION_90,
    HF_CAMERA_ROTATION_180,
    HF_CAMERA_ROTATION_270,
    HF_DETECT_MODE_ALWAYS_DETECT,
    HF_DETECT_MODE_LIGHT_TRACK,
    HF_ENABLE_NONE,
    HF_ENABLE_FACE_POSE,
)

from .common import (
    NativeResourceCaseMixin,
    calculate_iou,
    load_image,
    managed_session,
)


def restore_rotated_box(original_width, original_height, box, rotation):
    x1, y1, x2, y2 = box
    if rotation == 1:
        width = original_height
        return y1, width - x2, y2, width - x1
    if rotation == 2:
        return (
            original_width - x2,
            original_height - y2,
            original_width - x1,
            original_height - y1,
        )
    if rotation == 3:
        height = original_width
        return height - y2, x1, height - y1, x2
    return box


class FaceTrackCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_custom_parameter_exposes_cpp_face_pose_field(self):
        parameter = ifac.SessionCustomParameter(
            enable_detect_mode_landmark=True,
            enable_face_pose=True,
        )
        native = parameter._c_struct()
        self.assertEqual(native.enable_detect_mode_landmark, 1)
        self.assertEqual(native.enable_face_pose, 1)

    def test_face_and_no_face_detection(self):
        with managed_session(HF_ENABLE_NONE, max_detect_num=3) as session:
            face_image = load_image("bulk/kun.jpg")
            faces = session.face_detection(face_image)
            self.assertEqual(len(faces), 1)
            expected = (79, 104, 247, 271)
            self.assertGreater(calculate_iou(faces[0].location, expected), 0.5)
            self.assertGreaterEqual(faces[0].detection_confidence, 0.0)
            self.assertLessEqual(faces[0].detection_confidence, 1.0)
            self.assertGreaterEqual(faces[0].track_count, 0)

            no_face = load_image("bulk/view.jpg")
            self.assertEqual(session.face_detection(no_face), [])

    def test_rotated_detection_consistency(self):
        reference = load_image("rotate/rot_0.jpg")
        height, width = reference.shape[:2]
        with managed_session(HF_ENABLE_NONE) as session:
            reference_faces = session.face_detection(reference)
            self.assertGreater(len(reference_faces), 0)
            reference_box = reference_faces[0].location
            cases = (
                ("rotate/rot_90.jpg", HF_CAMERA_ROTATION_90),
                ("rotate/rot_180.jpg", HF_CAMERA_ROTATION_180),
                ("rotate/rot_270.jpg", HF_CAMERA_ROTATION_270),
            )
            for relative_path, rotation in cases:
                image = load_image(relative_path)
                with ifac.ImageStream.load_from_cv_image(
                    image,
                    rotation=rotation,
                ) as stream:
                    faces = session.face_detection(stream)
                self.assertEqual(len(faces), len(reference_faces), relative_path)
                restored = restore_rotated_box(width, height, faces[0].location, rotation)
                self.assertGreater(calculate_iou(restored, reference_box), 0.90, relative_path)

    def test_head_pose_signs_match_cpp_cases(self):
        cases = (
            ("pose/left_face.jpeg", "yaw", lambda value: -90 < value < -10),
            ("pose/right_face.png", "yaw", lambda value: 10 < value < 90),
            ("pose/rise_face.jpeg", "pitch", lambda value: value > 3),
            ("pose/lower_face.jpeg", "pitch", lambda value: value < -10),
            ("pose/left_wryneck.png", "roll", lambda value: value < -30),
            ("pose/right_wryneck.png", "roll", lambda value: value > 25),
        )
        with managed_session(HF_ENABLE_FACE_POSE) as session:
            for relative_path, attribute, predicate in cases:
                image = load_image(relative_path)
                faces = session.face_detection(image)
                self.assertEqual(len(faces), 1, relative_path)
                value = getattr(faces[0], attribute)
                self.assertTrue(predicate(value), "{} {}={}".format(relative_path, attribute, value))

    def test_landmark_shapes_and_values(self):
        with managed_session(HF_ENABLE_NONE) as session:
            image = load_image("bulk/kun.jpg")
            faces = session.face_detection(image)
            self.assertEqual(len(faces), 1)
            five = session.get_face_five_key_points(faces[0])
            dense = session.get_face_dense_landmark(faces[0])
            self.assertEqual(five.shape, (5, 2))
            self.assertEqual(dense.ndim, 2)
            self.assertEqual(dense.shape[1], 2)
            self.assertGreater(dense.shape[0], 5)
            self.assertTrue(np.isfinite(five).all())
            self.assertTrue(np.isfinite(dense).all())

    def test_light_track_id_stability(self):
        image = load_image("bulk/kun.jpg")
        with managed_session(
            HF_ENABLE_NONE,
            detect_mode=HF_DETECT_MODE_LIGHT_TRACK,
        ) as session:
            results = [session.face_detection(image) for _ in range(6)]
        self.assertTrue(all(len(faces) == 1 for faces in results))
        track_ids = [faces[0].track_id for faces in results]
        self.assertEqual(len(set(track_ids)), 1)

    def test_detection_pixel_levels_match_cpp_ranges(self):
        image = load_image("bulk/pedestrian.png")
        cases = (
            (192, 1, 6, 20),
            (320, 10, 11, 20),
            (640, 16, 20, 25),
        )
        for level, minimum, maximum, max_detect_num in cases:
            with managed_session(
                HF_ENABLE_NONE,
                detect_mode=HF_DETECT_MODE_ALWAYS_DETECT,
                max_detect_num=max_detect_num,
                detect_pixel_level=level,
            ) as session:
                session.set_track_preview_size(level)
                session.set_filter_minimum_face_pixel_size(0)
                actual_level = HInt32()
                check_error(
                    HFSessionGetTrackPreviewSize(
                        session._sess,
                        ctypes.byref(actual_level),
                    ),
                    "Get track preview size",
                )
                self.assertEqual(actual_level.value, level)
                count = len(session.face_detection(image))
                self.assertGreaterEqual(count, minimum, level)
                self.assertLessEqual(count, maximum, level)

    def test_invalid_preview_level_is_clamped(self):
        with managed_session(
            HF_ENABLE_NONE,
            max_detect_num=20,
            detect_pixel_level=1000,
        ) as session:
            actual_level = HInt32()
            check_error(
                HFSessionGetTrackPreviewSize(session._sess, ctypes.byref(actual_level)),
                "Get clamped preview size",
            )
            self.assertEqual(actual_level.value, 640)
            session.set_track_preview_size(192)
            check_error(
                HFSessionGetTrackPreviewSize(session._sess, ctypes.byref(actual_level)),
                "Get updated preview size",
            )
            self.assertEqual(actual_level.value, 192)
