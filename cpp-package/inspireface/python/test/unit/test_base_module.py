from test import *
import unittest
import inspireface as ifac
from inspireface.param import *
import cv2


class CameraStreamCase(unittest.TestCase):
    def setUp(self) -> None:
        """Shared area for priority execution"""
        pass

    def test_image_codec(self) -> None:
        image = cv2.imread(get_test_data("bulk/kun.jpg"))
        self.assertIsNotNone(image)

    def test_stream_rotation(self) -> None:
        # Prepare material
        engine = ifac.InspireFaceSession(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT)
        # Prepare rotation images
        rotation_images_filenames = ["rotate/rot_0.jpg", "rotate/rot_90.jpg", "rotate/rot_180.jpg","rotate/rot_270.jpg"]
        rotation_images = [cv2.imread(get_test_data(path)) for path in rotation_images_filenames]
        self.assertEqual(True, all(isinstance(item, np.ndarray) for item in rotation_images))

        # Detecting face images without rotation
        rot_0 = rotation_images[0]
        h, w, _ = rot_0.shape
        self.assertIsNotNone(rot_0, "Image is empty")
        rot_0_faces = engine.face_detection(image=rot_0)
        self.assertEqual(True, len(rot_0_faces) > 0)
        rot_0_face_box = rot_0_faces[0].location
        num_of_faces = len(rot_0_faces)

        # Detect images with other rotation angles
        rotation_tags = [HF_CAMERA_ROTATION_90, HF_CAMERA_ROTATION_180, HF_CAMERA_ROTATION_270]
        streams = [ifac.ImageStream.load_from_cv_image(img, rotation=rotation_tags[idx]) for idx, img in enumerate(rotation_images[1:])]
        results = [engine.face_detection(stream) for stream in streams]
        # No matter how many degrees the image is rotated, the same number of faces should be detected
        self.assertEqual(True, all(len(item) == num_of_faces for item in results))
        # Select all the first face box
        rot_other_faces_boxes = [face[0].location for face in results]
        # We need to restore the rotated face box
        restored_boxes = [restore_rotated_box(w, h, rot_other_faces_boxes[idx], rotation_tags[idx]) for idx, box in enumerate(rot_other_faces_boxes)]
        # IoU is performed with the face box of the original image to calculate the overlap
        iou_results = [calculate_overlap(box, rot_0_face_box) for box in restored_boxes]
        # The face box position of all rotated images is detected to be consistent with that of the original image
        self.assertEqual(all(0.95 < iou < 1.0 for iou in iou_results), True)


if __name__ == '__main__':
    unittest.main()
