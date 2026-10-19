"""Face pipeline cases ported from C++ test_face_pipeline.cpp."""

import unittest

import inspireface as ifac
from inspireface.modules.core import HFFaceMaskConfidence
from inspireface.modules.exception import ProcessingError
from inspireface.param import (
    HF_ENABLE_FACE_ATTRIBUTE,
    HF_ENABLE_FACE_EMOTION,
    HF_ENABLE_INTERACTION,
    HF_ENABLE_LIVENESS,
    HF_ENABLE_MASK_DETECT,
    HF_ENABLE_QUALITY,
)

from .common import NativeResourceCaseMixin, load_image, managed_session


def pipeline_fixture(testcase, relative_path, parameter):
    image = load_image(relative_path)
    with managed_session(parameter, max_detect_num=5, detect_pixel_level=320) as session:
        faces = session.face_detection(image)
        testcase.assertGreater(len(faces), 0, relative_path)
        extended = session.face_pipeline(image, faces, parameter)
        testcase.assertEqual(len(extended), len(faces), relative_path)
        return extended


class FacePipelineCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_pipeline_result_count_mismatch_is_rejected(self):
        result = HFFaceMaskConfidence()
        result.num = 2
        with self.assertRaises(ProcessingError):
            ifac.InspireFaceSession._validate_pipeline_result(
                0,
                result,
                ("confidence",),
                1,
                "Synthetic mask result",
            )
        result.num = 1
        with self.assertRaises(ProcessingError):
            ifac.InspireFaceSession._validate_pipeline_result(
                0,
                result,
                ("confidence",),
                1,
                "Synthetic mask result",
            )

    def test_empty_pipeline_is_a_no_op(self):
        with managed_session(0) as session:
            self.assertEqual(session.face_pipeline(object(), [], 0), [])

    def test_emotion_classification_matches_cpp_cases(self):
        cases = (
            ("emotion/anger.png", 6),
            ("emotion/sad.png", 2),
            ("emotion/happy.png", 1),
        )
        for relative_path, expected_emotion in cases:
            results = pipeline_fixture(self, relative_path, HF_ENABLE_FACE_EMOTION)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].emotion, expected_emotion, relative_path)

    def test_face_attributes_match_cpp_cases(self):
        parameter = ifac.SessionCustomParameter(enable_face_attribute=True)
        black_girl = pipeline_fixture(self, "attribute/1423.jpg", parameter)
        self.assertEqual(len(black_girl), 1)
        self.assertEqual(black_girl[0].race, 0)
        self.assertEqual(black_girl[0].age_bracket, 2)
        self.assertIn(black_girl[0].gender, range(2))

        women = pipeline_fixture(self, "attribute/7242.jpg", HF_ENABLE_FACE_ATTRIBUTE)
        self.assertEqual(len(women), 2)
        for result in women:
            self.assertEqual(result.race, 4)
            self.assertIn(result.age_bracket, range(9))
            self.assertEqual(result.gender, 0)

    def test_mask_confidence_matches_cpp_cases(self):
        parameter = ifac.SessionCustomParameter(enable_mask_detect=True)
        masked = pipeline_fixture(self, "bulk/mask2.jpg", parameter)
        unmasked = pipeline_fixture(self, "bulk/face_sample.png", parameter)
        self.assertGreater(masked[0].mask_confidence, 0.9)
        self.assertLess(unmasked[0].mask_confidence, 0.1)

    def test_rgb_liveness_real_and_fake(self):
        parameter = ifac.SessionCustomParameter(enable_liveness=True)
        real = pipeline_fixture(self, "bulk/image_T1.jpeg", parameter)
        fake = pipeline_fixture(self, "bulk/rgb_fake.jpg", parameter)
        self.assertGreaterEqual(real[0].rgb_liveness_confidence, 0.0)
        self.assertLessEqual(real[0].rgb_liveness_confidence, 1.0)
        self.assertGreaterEqual(fake[0].rgb_liveness_confidence, 0.0)
        self.assertLess(fake[0].rgb_liveness_confidence, 0.9)
        self.assertGreater(real[0].rgb_liveness_confidence, fake[0].rgb_liveness_confidence)

    def test_face_quality_orders_clear_above_blur(self):
        clear = pipeline_fixture(self, "bulk/yifei.jpg", HF_ENABLE_QUALITY)
        blur = pipeline_fixture(self, "bulk/blur.jpg", HF_ENABLE_QUALITY)
        self.assertGreaterEqual(clear[0].quality_confidence, 0.0)
        self.assertLessEqual(clear[0].quality_confidence, 1.0)
        self.assertGreaterEqual(blur[0].quality_confidence, 0.0)
        self.assertLessEqual(blur[0].quality_confidence, 1.0)
        self.assertGreater(clear[0].quality_confidence, blur[0].quality_confidence)

    def test_eye_state_matches_cpp_cases(self):
        parameter = ifac.SessionCustomParameter(
            enable_interaction_liveness=True,
            enable_liveness=True,
        )
        cases = (
            ("reaction/open_eyes.png", lambda left, right: left > 0.5 and right > 0.5),
            ("reaction/close_eyes.jpeg", lambda left, right: left < 0.5 and right < 0.5),
            ("reaction/close_open_eyes.jpeg", lambda left, right: left < 0.5 and right > 0.5),
        )
        for relative_path, predicate in cases:
            result = pipeline_fixture(self, relative_path, parameter)[0]
            self.assertTrue(
                predicate(
                    result.left_eye_status_confidence,
                    result.right_eye_status_confidence,
                ),
                "{} left={} right={}".format(
                    relative_path,
                    result.left_eye_status_confidence,
                    result.right_eye_status_confidence,
                ),
            )

    def test_combined_pipeline_populates_enabled_fields(self):
        parameter = ifac.SessionCustomParameter(
            enable_recognition=True,
            enable_liveness=True,
            enable_mask_detect=True,
            enable_face_attribute=True,
            enable_face_quality=True,
            enable_interaction_liveness=True,
            enable_face_emotion=True,
        )
        result = pipeline_fixture(self, "reaction/open_eyes.png", parameter)[0]
        self.assertGreaterEqual(result.rgb_liveness_confidence, 0.0)
        self.assertGreaterEqual(result.mask_confidence, 0.0)
        self.assertGreaterEqual(result.quality_confidence, 0.0)
        self.assertGreaterEqual(result.left_eye_status_confidence, 0.0)
        self.assertGreaterEqual(result.right_eye_status_confidence, 0.0)
        self.assertIn(result.race, range(5))
        self.assertIn(result.gender, range(2))
        self.assertIn(result.age_bracket, range(9))
        self.assertIn(result.emotion, range(7))

    def test_optional_bitmask_pipeline(self):
        option = HF_ENABLE_MASK_DETECT | HF_ENABLE_QUALITY | HF_ENABLE_INTERACTION
        results = pipeline_fixture(self, "reaction/open_eyes.png", option)
        self.assertGreater(results[0].mask_confidence, -1.0)
        self.assertGreater(results[0].quality_confidence, -1.0)
        self.assertGreater(results[0].left_eye_status_confidence, -1.0)
