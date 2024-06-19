import unittest
from test import *
import inspireface as ifac
from inspireface.param import *
import cv2


class FaceRecognitionBaseCase(unittest.TestCase):
    """
    This case is mainly used to test the basic functions of face recognition.
    """

    def setUp(self) -> None:
        # Prepare material
        track_mode = HF_DETECT_MODE_ALWAYS_DETECT
        param = ifac.SessionCustomParameter()
        param.enable_recognition = True
        self.engine = ifac.InspireFaceSession(param, track_mode, 10)

    def test_face_feature_extraction(self):
        # Prepare a image
        image = cv2.imread(get_test_data("bulk/kun.jpg"))
        self.assertIsNotNone(image)
        # Face detection
        faces = self.engine.face_detection(image)
        # "kun.jpg" has only one face
        self.assertEqual(len(faces), 1)
        face = faces[0]
        box = face.location
        expect_box = (98, 146, 233, 272)
        # Calculate the location of the detected box and the expected box
        iou = calculate_overlap(box, expect_box)
        self.assertAlmostEqual(iou, 1.0, places=3)

        # Extract feature
        feature = self.engine.face_feature_extract(image, face)
        self.assertIsNotNone(feature)
#
    def test_face_comparison(self):
        # Prepare two pictures of someone
        images_path_list = [get_test_data("bulk/kun.jpg"), get_test_data("bulk/jntm.jpg")]
        self.assertEqual(len(images_path_list), 2, "Only 2 photos can be used for the 1v1 scene.")
        images = [cv2.imread(pth) for pth in images_path_list]
        faces_list = [self.engine.face_detection(img) for img in images]
        # Check num of faces detection
        self.assertEqual(len(faces_list[0]), 1)
        self.assertEqual(len(faces_list[1]), 1)
        # Extract features
        features = [self.engine.face_feature_extract(images[idx], faces[0]) for idx, faces in enumerate(faces_list)]
        self.assertEqual(features[0].size, TEST_MODEL_FACE_FEATURE_LENGTH)
        self.assertEqual(features[1].size, TEST_MODEL_FACE_FEATURE_LENGTH)
        # Comparison
        similarity = ifac.feature_comparison(features[0], features[1])
        self.assertEqual(True, similarity > TEST_FACE_COMPARISON_IMAGE_THRESHOLD)

        # Prepare a picture of a different person
        woman = cv2.imread(get_test_data("bulk/woman.png"))
        self.assertIsNotNone(woman)
        woman_faces = self.engine.face_detection(woman)
        self.assertEqual(len(woman_faces), 1)
        face_3 = woman_faces[0]
        feature = self.engine.face_feature_extract(woman, face_3)
        self.assertEqual(feature.size, TEST_MODEL_FACE_FEATURE_LENGTH)
        # Comparison
        similarity = ifac.feature_comparison(features[0], feature)
        self.assertEqual(True, similarity < TEST_FACE_COMPARISON_IMAGE_THRESHOLD)
        similarity = ifac.feature_comparison(features[1], feature)
        self.assertEqual(True, similarity < TEST_FACE_COMPARISON_IMAGE_THRESHOLD)


@optional(ENABLE_CRUD_TEST, "All CRUD related tests have been closed.")
class FaceRecognitionCRUDMemoryCase(unittest.TestCase):
    """
    This case is mainly used to test the CRUD functions of face recognition.
    """

    engine = None
    default_faces_num = 10000

    @classmethod
    def setUpClass(cls):
        config = ifac.FeatureHubConfiguration(
            feature_block_num=20,
            enable_use_db=False,
            db_path="",
            search_mode=HF_SEARCH_MODE_EAGER,
            search_threshold=TEST_FACE_COMPARISON_IMAGE_THRESHOLD,
        )
        ifac.feature_hub_enable(config)
        track_mode = HF_DETECT_MODE_ALWAYS_DETECT
        param = ifac.SessionCustomParameter()
        param.enable_recognition = True
        cls.engine = ifac.InspireFaceSession(param, track_mode)
        batch_import_lfw_faces(LFW_FUNNELED_DIR_PATH, cls.engine, cls.default_faces_num)


    def test_face_search(self):
        num_current = ifac.feature_hub_get_face_count()
        registered = cv2.imread(get_test_data("bulk/kun.jpg"))
        self.assertIsNotNone(registered)
        faces = self.engine.face_detection(registered)
        self.assertEqual(len(faces), 1)
        face = faces[0]
        feature = self.engine.face_feature_extract(registered, face)
        self.assertEqual(feature.size, TEST_MODEL_FACE_FEATURE_LENGTH)
        # Insert a new face
        registered_identity = ifac.FaceIdentity(feature, custom_id=num_current + 1, tag="Kun")
        ret = ifac.feature_hub_face_insert(registered_identity)
        self.assertEqual(ret, True)

        # Prepare a picture of searched face
        searched = cv2.imread(get_test_data("bulk/jntm.jpg"))
        self.assertIsNotNone(searched)
        faces = self.engine.face_detection(searched)
        self.assertEqual(len(faces), 1)
        searched_face = faces[0]
        feature = self.engine.face_feature_extract(searched, searched_face)
        self.assertEqual(feature.size, TEST_MODEL_FACE_FEATURE_LENGTH)
        searched_result = ifac.feature_hub_face_search(feature)
        self.assertEqual(True, searched_result.confidence > TEST_FACE_COMPARISON_IMAGE_THRESHOLD)
        self.assertEqual(searched_result.similar_identity.tag, registered_identity.tag)
        self.assertEqual(searched_result.similar_identity.custom_id, registered_identity.custom_id)

        # Prepare a picture of a stranger's face
        stranger = cv2.imread(get_test_data("bulk/woman.png"))
        self.assertIsNotNone(stranger)
        faces = self.engine.face_detection(stranger)
        self.assertEqual(len(faces), 1)
        stranger_face = faces[0]
        feature = self.engine.face_feature_extract(stranger, stranger_face)
        self.assertEqual(feature.size, TEST_MODEL_FACE_FEATURE_LENGTH)
        stranger_result = ifac.feature_hub_face_search(feature)
        self.assertEqual(True, stranger_result.confidence < TEST_FACE_COMPARISON_IMAGE_THRESHOLD)
        self.assertEqual(stranger_result.similar_identity.custom_id, -1)
#
    def test_face_remove(self):
        query_image = cv2.imread(get_test_data("bulk/Nathalie_Baye_0002.jpg"))
        self.assertIsNotNone(query_image)
        faces = self.engine.face_detection(query_image)
        self.assertEqual(len(faces), 1)
        query_face = faces[0]
        feature = self.engine.face_feature_extract(query_image, query_face)
        self.assertEqual(feature.size, TEST_MODEL_FACE_FEATURE_LENGTH)
        # First search
        result = ifac.feature_hub_face_search(feature)
        self.assertEqual(True, result.confidence > TEST_FACE_COMPARISON_IMAGE_THRESHOLD)
        self.assertEqual("Nathalie_Baye", result.similar_identity.tag)

        # Remove that
        remove_id = result.similar_identity.custom_id
        ret = ifac.feature_hub_face_remove(remove_id)
        self.assertEqual(ret, True)

        # Second search
        result = ifac.feature_hub_face_search(feature)
        self.assertEqual(True, result.confidence < TEST_FACE_COMPARISON_IMAGE_THRESHOLD)
        self.assertEqual(result.similar_identity.custom_id, -1)

        # Reusability testing
        new_face_image = cv2.imread(get_test_data("bulk/yifei.jpg"))
        self.assertIsNotNone(new_face_image)
        faces = self.engine.face_detection(new_face_image)
        self.assertEqual(len(faces), 1)
        new_face = faces[0]
        feature = self.engine.face_feature_extract(new_face_image, new_face)
        # Insert that
        registered_identity = ifac.FaceIdentity(feature, custom_id=remove_id, tag="YF")
        ifac.feature_hub_face_insert(registered_identity)

    def test_face_update(self):
        pass


@optional(ENABLE_BENCHMARK_TEST, "All benchmark related tests have been closed.")
class FaceRecognitionFeatureExtractCase(unittest.TestCase):
    benchmark_results = list()
    loop = 1

    @classmethod
    def setUpClass(cls):
        cls.benchmark_results = []

    def setUp(self) -> None:
        # Prepare image
        image = cv2.imread(get_test_data("bulk/kun.jpg"))
        self.stream = ifac.ImageStream.load_from_cv_image(image)
        self.assertIsNotNone(self.stream)
        # Prepare material
        track_mode = HF_DETECT_MODE_ALWAYS_DETECT
        param = ifac.SessionCustomParameter()
        param.enable_recognition = True
        self.engine = ifac.InspireFaceSession(param, track_mode)
        # Prepare a face
        faces = self.engine.face_detection(self.stream)
        # "kun.jpg" has only one face
        self.assertEqual(len(faces), 1)
        self.face = faces[0]
        box = self.face.location
        expect_box = (98, 146, 233, 272)
        # Calculate the location of the detected box and the expected box
        iou = calculate_overlap(box, expect_box)
        self.assertAlmostEqual(iou, 1.0, places=3)
        self.feature = self.engine.face_feature_extract(self.stream, self.face)


    @benchmark(test_name="Feature Extract", loop=1000)
    def test_benchmark_feature_extract(self):
        for _ in range(self.loop):
            feature = self.engine.face_feature_extract(self.stream, self.face)
            self.assertEqual(TEST_MODEL_FACE_FEATURE_LENGTH, feature.size)

    @benchmark(test_name="Face comparison 1v1", loop=1000)
    def test_benchmark_face_comparison1v1(self):
        for _ in range(self.loop):
            ifac.feature_comparison(self.feature, self.feature)

    @classmethod
    def tearDownClass(cls):
        print_benchmark_table(cls.benchmark_results)


if __name__ == '__main__':
    unittest.main()
