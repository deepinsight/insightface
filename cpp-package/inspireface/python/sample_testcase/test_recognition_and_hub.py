"""Recognition, parallel-session, and FeatureHub regression cases."""

import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import inspireface as ifac
from inspireface.param import (
    HF_PK_AUTO_INCREMENT,
    HF_PK_MANUAL_INPUT,
    HF_INVALID_FACE_ID,
    HF_SEARCH_MODE_EXHAUSTIVE,
)

from .common import (
    NativeResourceCaseMixin,
    extract_first_feature,
    managed_feature_hub,
    managed_session,
    record_metric,
)


def recognition_parameter():
    return ifac.SessionCustomParameter(enable_recognition=True)


def three_features():
    with managed_session(recognition_parameter()) as session:
        return (
            extract_first_feature(session, "bulk/kun.jpg"),
            extract_first_feature(session, "bulk/jntm.jpg"),
            extract_first_feature(session, "bulk/yifei.jpg"),
        )


class RecognitionCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_feature_extract_and_comparison(self):
        with managed_session(recognition_parameter()) as session:
            feature = extract_first_feature(session, "bulk/kun.jpg")
            same_person = extract_first_feature(session, "bulk/kun_crop.jpg")
            different_person = extract_first_feature(session, "bulk/jntm.jpg")
        self.assertEqual(feature.dtype, np.float32)
        self.assertEqual(feature.ndim, 1)
        self.assertEqual(feature.size, 512)
        self.assertTrue(np.isfinite(feature).all())

        self_similarity = ifac.feature_comparison(feature, feature)
        same_similarity = ifac.feature_comparison(feature, same_person)
        different_similarity = ifac.feature_comparison(feature, different_person)
        self.assertGreater(self_similarity, 0.99)
        self.assertGreater(same_similarity, different_similarity)
        self.assertGreater(same_similarity, ifac.get_recommended_cosine_threshold())

    def test_parallel_sessions_are_deterministic(self):
        def compare_once(_):
            with managed_session(recognition_parameter()) as session:
                first = extract_first_feature(session, "bulk/kun.jpg")
                second = extract_first_feature(session, "bulk/jntm.jpg")
                return ifac.feature_comparison(first, second)

        with ThreadPoolExecutor(max_workers=4) as executor:
            similarities = list(executor.map(compare_once, range(8)))
        baseline = similarities[0]
        for similarity in similarities[1:]:
            self.assertAlmostEqual(similarity, baseline, delta=0.01)


class FeatureHubCase(NativeResourceCaseMixin, unittest.TestCase):
    def memory_configuration(self, primary_key_mode=HF_PK_MANUAL_INPUT):
        return ifac.FeatureHubConfiguration(
            primary_key_mode=primary_key_mode,
            enable_persistence=False,
            persistence_db_path="",
            search_threshold=0.35,
            search_mode=HF_SEARCH_MODE_EXHAUSTIVE,
        )

    def test_manual_crud_search_and_top_k(self):
        first, second, third = three_features()
        configuration = self.memory_configuration()
        with managed_feature_hub(configuration):
            ifac.feature_hub_set_search_threshold(0.35)
            inserted, allocated = ifac.feature_hub_face_insert(ifac.FaceIdentity(first, 101))
            self.assertTrue(inserted)
            self.assertEqual(allocated, 101)
            ifac.feature_hub_face_insert(ifac.FaceIdentity(second, 202))
            self.assertEqual(ifac.feature_hub_get_face_count(), 2)
            self.assertEqual(set(ifac.feature_hub_get_face_id_list()), {101, 202})

            identity = ifac.feature_hub_get_face_identity(101)
            self.assertEqual(identity.id, 101)
            np.testing.assert_allclose(identity.feature, first, rtol=0.0, atol=1e-6)
            identity_snapshot = identity.feature.copy()

            result = ifac.feature_hub_face_search(first)
            self.assertTrue(result.matched)
            self.assertEqual(result.similar_identity.id, 101)
            self.assertGreater(result.confidence, 0.9)
            top_k = ifac.feature_hub_face_search_top_k(first, 2)
            self.assertEqual(len(top_k), 2)
            self.assertEqual(top_k[0][1], 101)

            ifac.feature_hub_set_search_threshold(1.0)
            no_match = ifac.feature_hub_face_search(third)
            self.assertFalse(no_match.matched)
            self.assertEqual(no_match.similar_identity.id, HF_INVALID_FACE_ID)
            self.assertEqual(no_match.similar_identity.feature.size, 0)
            self.assertEqual(no_match.confidence, -1.0)
            ifac.feature_hub_set_search_threshold(0.35)

            self.assertTrue(ifac.feature_hub_face_update(ifac.FaceIdentity(third, 101)))
            updated = ifac.feature_hub_face_search(third)
            self.assertEqual(updated.similar_identity.id, 101)
            np.testing.assert_array_equal(identity.feature, identity_snapshot)
            self.assertTrue(ifac.feature_hub_face_remove(202))
            self.assertEqual(ifac.feature_hub_get_face_count(), 1)

    def test_face_identity_owns_feature_memory(self):
        feature = np.arange(512, dtype=np.float32)
        identity = ifac.FaceIdentity(feature, 303)
        feature[:] = -1.0
        self.assertEqual(identity.feature[10], 10.0)

    def test_auto_increment_allocates_unique_ids(self):
        first, second, _ = three_features()
        with managed_feature_hub(self.memory_configuration(HF_PK_AUTO_INCREMENT)):
            _, first_id = ifac.feature_hub_face_insert(ifac.FaceIdentity(first, HF_INVALID_FACE_ID))
            _, second_id = ifac.feature_hub_face_insert(ifac.FaceIdentity(second, HF_INVALID_FACE_ID))
            self.assertGreaterEqual(first_id, 0)
            self.assertGreater(second_id, first_id)
            self.assertEqual(ifac.feature_hub_get_face_count(), 2)

    def test_manual_mode_reserves_invalid_id_and_supports_64_bit_ids(self):
        first, second, _ = three_features()
        wide_id = (1 << 40) + 17
        with managed_feature_hub(self.memory_configuration()):
            with self.assertRaises(ifac.InvalidInputError):
                ifac.feature_hub_face_insert(ifac.FaceIdentity(first, HF_INVALID_FACE_ID))

            inserted, allocated = ifac.feature_hub_face_insert(ifac.FaceIdentity(first, wide_id))
            self.assertTrue(inserted)
            self.assertEqual(allocated, wide_id)
            identity = ifac.feature_hub_get_face_identity(wide_id)
            self.assertEqual(identity.id, wide_id)
            np.testing.assert_allclose(identity.feature, first, rtol=0.0, atol=1e-6)

            result = ifac.feature_hub_face_search(first)
            self.assertTrue(result.matched)
            self.assertEqual(result.similar_identity.id, wide_id)
            self.assertGreater(result.confidence, 0.99)
            self.assertEqual(ifac.feature_hub_face_search_top_k(first, 1)[0][1], wide_id)

            self.assertTrue(ifac.feature_hub_face_update(ifac.FaceIdentity(second, wide_id)))
            updated = ifac.feature_hub_get_face_identity(wide_id)
            np.testing.assert_allclose(updated.feature, second, rtol=0.0, atol=1e-6)
            self.assertTrue(ifac.feature_hub_face_remove(wide_id))
            self.assertEqual(ifac.feature_hub_get_face_count(), 0)

    def test_persistent_mode_survives_reenable(self):
        first, _, _ = three_features()
        persistent_id = (1 << 40) + 501
        with tempfile.TemporaryDirectory(prefix="inspireface-feature-hub-") as temp_dir:
            database = Path(temp_dir) / "feature.db"
            configuration = ifac.FeatureHubConfiguration(
                primary_key_mode=HF_PK_MANUAL_INPUT,
                enable_persistence=True,
                persistence_db_path=str(database),
                search_threshold=0.35,
                search_mode=HF_SEARCH_MODE_EXHAUSTIVE,
            )
            ifac.feature_hub_enable(configuration)
            try:
                ifac.feature_hub_face_insert(ifac.FaceIdentity(first, persistent_id))
                self.assertEqual(ifac.feature_hub_get_face_count(), 1)
            finally:
                ifac.feature_hub_disable()

            ifac.feature_hub_enable(configuration)
            try:
                self.assertEqual(ifac.feature_hub_get_face_count(), 1)
                self.assertEqual(ifac.feature_hub_get_face_identity(persistent_id).id, persistent_id)
                result = ifac.feature_hub_face_search(first)
                self.assertTrue(result.matched)
                self.assertEqual(result.similar_identity.id, persistent_id)
            finally:
                ifac.feature_hub_disable()

    def test_parallel_search_is_stable(self):
        first, second, _ = three_features()
        with managed_feature_hub(self.memory_configuration()):
            ifac.feature_hub_face_insert(ifac.FaceIdentity(first, 701))
            ifac.feature_hub_face_insert(ifac.FaceIdentity(second, 702))

            def search(_):
                result = ifac.feature_hub_face_search(first)
                return result.similar_identity.id, result.confidence

            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(search, range(32)))
            elapsed = time.perf_counter() - started
            record_metric("feature_hub_parallel_search_32_total_ms", elapsed * 1000.0)
            self.assertTrue(all(identity_id == 701 for identity_id, _ in results))
            baseline = results[0][1]
            for _, confidence in results:
                self.assertAlmostEqual(confidence, baseline, delta=1e-6)
            self.assertLess(elapsed, 2.0)
