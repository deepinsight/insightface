import unittest
from test import *
import inspireface as ifac
from inspireface.param import *
import cv2


class FaceTrackerCase(unittest.TestCase):

    def setUp(self) -> None:
        # Prepare material
        track_mode = HF_DETECT_MODE_IMAGE  # Use video mode
        self.engine = ifac.InspireFaceSession(param=ifac.SessionCustomParameter(),
                                        detect_mode=track_mode)

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
        self.engine.set_track_mode(HF_DETECT_MODE_IMAGE)

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

    def test_face_track_from_video(self):
        self.engine.set_track_mode(HF_DETECT_MODE_VIDEO)

        # Read a video file
        video_gen = read_video_generator(get_test_data("video/810_1684206192.mp4"))
        results = [self.engine.face_detection(frame) for frame in video_gen]
        num_of_frame = len(results)
        num_of_track_loss = len([faces for faces in results if not faces])
        total_track_ids = [faces[0].track_id for faces in results if faces]
        num_of_id_switch = len([id_ for id_ in total_track_ids if id_ != 1])

        # Calculate the loss rate of trace loss and switching id
        track_loss = num_of_track_loss / num_of_frame
        id_switch_loss = num_of_id_switch / len(total_track_ids)

        # Not rigorous, only for the current test of this video file
        self.assertEqual(True, track_loss < 0.05)
        self.assertEqual(True, id_switch_loss < 0.1)


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
        track_mode = HF_DETECT_MODE_VIDEO  # Use video mode
        self.engine = ifac.InspireFaceSession(HF_ENABLE_NONE, track_mode, )
        # Prepare video data
        self.video_gen = read_video_generator(get_test_data("video/810_1684206192.mp4"))

    @benchmark(test_name="Face Detect", loop=1000)
    def test_benchmark_face_detect(self):
        self.engine.set_track_mode(HF_DETECT_MODE_IMAGE)
        for _ in range(self.loop):
            faces = self.engine.face_detection(self.image)
            self.assertEqual(len(faces), 1, "No face detected may have an error, please check.")

    @benchmark(test_name="Face Track", loop=1000)
    def test_benchmark_face_track(self):
        self.engine.set_track_mode(HF_DETECT_MODE_VIDEO)
        for _ in range(self.loop):
            faces = self.engine.face_detection(self.image)
            self.assertEqual(len(faces), 1, "No face detected may have an error, please check.")

    @benchmark(test_name="Face Track(Video)", loop=345)
    def test_benchmark_face_track_video(self):
        self.engine.set_track_mode(HF_DETECT_MODE_VIDEO)
        for frame in self.video_gen:
            faces = self.engine.face_detection(frame)
            self.assertEqual(len(faces), 1, "No face detected may have an error, please check.")

    @classmethod
    def tearDownClass(cls):
        print_benchmark_table(cls.benchmark_results)


if __name__ == '__main__':
    unittest.main()
