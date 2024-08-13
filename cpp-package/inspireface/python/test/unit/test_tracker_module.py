import unittest
from test import *
import inspireface as ifac
from inspireface.param import *
import cv2


class FaceTrackerCase(unittest.TestCase):

    def setUp(self) -> None:
        # Prepare material
        track_mode = HF_DETECT_MODE_ALWAYS_DETECT
        self.engine = ifac.InspireFaceSession(param=ifac.SessionCustomParameter(),
                                        detect_mode=track_mode)

        self.engine_tk = ifac.InspireFaceSession(param=ifac.SessionCustomParameter(),
                                        detect_mode=HF_DETECT_MODE_LIGHT_TRACK)

    def test_face_detection_from_image(self):
        image = cv2.imread(get_test_data("bulk/kun.jpg"))
        self.assertIsNotNone(image)

        # Detection
        faces = self.engine.face_detection(image)
        # "kun.jpg" has only one face
        self.assertEqual(len(faces), 1)
        face = faces[0]
        expect_box = (98, 146, 233, 272)
        # Calculate the location of the detected box and the expected box
        iou = calculate_overlap(face.location, expect_box)
        self.assertAlmostEqual(iou, 1.0, places=3)

        # Prepare non-face images
        any_image = cv2.imread(get_test_data("bulk/view.jpg"))
        self.assertIsNotNone(any_image)
        self.assertEqual(len(self.engine.face_detection(any_image)), 0)

    def test_face_pose(self):
        # Test yaw (shake one's head)
        left_face = cv2.imread(get_test_data("pose/left_face.jpeg"))
        self.assertIsNotNone(left_face)
        faces = self.engine.face_detection(left_face)
        self.assertEqual(len(faces), 1)
        left_face_yaw = faces[0].yaw
        # The expected value is not completely accurate, it is only a rough estimate
        expect_left_shake_range = (-90, -10)
        self.assertEqual(True, expect_left_shake_range[0] < left_face_yaw < expect_left_shake_range[1])

        right_face = cv2.imread(get_test_data("pose/right_face.png"))
        self.assertIsNotNone(right_face)
        faces = self.engine.face_detection(right_face)
        self.assertEqual(len(faces), 1)
        right_face_yaw = faces[0].yaw
        expect_right_shake_range = (10, 90)
        self.assertEqual(True, expect_right_shake_range[0] < right_face_yaw < expect_right_shake_range[1])

        # Test pitch (nod head)
        rise_face = cv2.imread(get_test_data("pose/rise_face.jpeg"))
        self.assertIsNotNone(rise_face)
        faces = self.engine.face_detection(rise_face)
        self.assertEqual(len(faces), 1)
        left_face_pitch = faces[0].pitch
        self.assertEqual(True, left_face_pitch > 5)

        lower_face = cv2.imread(get_test_data("pose/lower_face.jpeg"))
        self.assertIsNotNone(lower_face)
        faces = self.engine.face_detection(lower_face)
        self.assertEqual(len(faces), 1)
        lower_face_pitch = faces[0].pitch
        self.assertEqual(True, lower_face_pitch < -10)

        # Test roll (wryneck head)
        left_wryneck_face = cv2.imread(get_test_data("pose/left_wryneck.png"))
        self.assertIsNotNone(left_wryneck_face)
        faces = self.engine.face_detection(left_wryneck_face)
        self.assertEqual(len(faces), 1)
        left_face_roll = faces[0].roll
        self.assertEqual(True, left_face_roll < -30)

        right_wryneck_face = cv2.imread(get_test_data("pose/right_wryneck.png"))
        self.assertIsNotNone(right_wryneck_face)
        faces = self.engine.face_detection(right_wryneck_face)
        self.assertEqual(len(faces), 1)
        right_face_roll = faces[0].roll
        self.assertEqual(True, right_face_roll > 30)

@optional(ENABLE_BENCHMARK_TEST, "All benchmark related tests have been closed.")
class FaceTrackerBenchmarkCase(unittest.TestCase):
    benchmark_results = list()
    loop = 1

    @classmethod
    def setUpClass(cls):
        cls.benchmark_results = []

    def setUp(self) -> None:
        # Prepare image
        self.image = cv2.imread(get_test_data("bulk/kun.jpg"))
        self.assertIsNotNone(self.image)
        # Prepare material
        self.engine = ifac.InspireFaceSession(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, )
        self.engine_tk = ifac.InspireFaceSession(HF_ENABLE_NONE, HF_DETECT_MODE_LIGHT_TRACK, )
        # Prepare video data
        self.video_gen = read_video_generator(get_test_data("video/810_1684206192.mp4"))

    @benchmark(test_name="Face Detect", loop=1000)
    def test_benchmark_face_detect(self):
        for _ in range(self.loop):
            faces = self.engine.face_detection(self.image)
            self.assertEqual(len(faces), 1, "No face detected may have an error, please check.")

    @benchmark(test_name="Face Track", loop=1000)
    def test_benchmark_face_track(self):
        for _ in range(self.loop):
            faces = self.engine_tk.face_detection(self.image)
            self.assertEqual(len(faces), 1, "No face detected may have an error, please check.")

    @benchmark(test_name="Face Track(Video)", loop=345)
    def test_benchmark_face_track_video(self):
        for frame in self.video_gen:
            faces = self.engine_tk.face_detection(frame)
            self.assertEqual(len(faces), 1, "No face detected may have an error, please check.")

    @classmethod
    def tearDownClass(cls):
        print_benchmark_table(cls.benchmark_results)


if __name__ == '__main__':
    unittest.main()
