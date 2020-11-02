from datetime import datetime

import cv2
import numpy as np
from retinaface import RetinaFace

# Drawing attributes
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
CIRCLE_RADIUS = 1
LINE_THICKNESS = 2

EYE_RIGHT_LANDMARK = 0
EYE_LEFT_LANDMARK = 1
NOSE_TIP_LANDMARK = 2
MOUTH_RIGHT_LANDMARK = 3
MOUTH_LEFT_LANDMARK = 4

landmark_to_color = {
    NOSE_TIP_LANDMARK: COLOR_RED,
    EYE_RIGHT_LANDMARK: COLOR_GREEN,
    MOUTH_RIGHT_LANDMARK: COLOR_GREEN,
    EYE_LEFT_LANDMARK: COLOR_BLUE,
    MOUTH_LEFT_LANDMARK: COLOR_BLUE
}

# Model attributes
THRESHOLD = 0.5
default_scale = [1024, 1980]
USE_GPU = 0
USE_CPU = -1
FLIP = False

detector = RetinaFace('./model/R50', 0, USE_CPU, 'net3')


def read_image_file(image_file_name):
    return cv2.imread(image_file_name)


def calculate_correct_image_scale(image, target_size, max_size):
    im_shape = image.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    scale_factor = float(target_size) / float(im_size_min)

    # prevent bigger axis from being more than max_size:
    if np.round(scale_factor * im_size_max) > max_size:
        scale_factor = float(max_size) / float(im_size_max)

    return [scale_factor]


def detect_face(image, scale):
    print("Start face detection")
    start = datetime.now()
    faces, landmarks = detector.detect(image, THRESHOLD, scales=scale, do_flip=FLIP)
    end = datetime.now()
    diff = end - start
    print('Detecting faces took: %s seconds' % diff.total_seconds())
    return [faces, landmarks]


def draw_faces_and_landmarks(faces, landmarks, output_file_name):
    if faces is not None:
        number_of_detected_faces = faces.shape[0]
        print('Found %s faces' % number_of_detected_faces)

        for face_index in range(number_of_detected_faces):
            box = faces[face_index].astype(np.int)

            top_left_point = (box[0], box[1])
            bottom_right_point = (box[2], box[3])
            cv2.rectangle(img, top_left_point, bottom_right_point, COLOR_RED, LINE_THICKNESS)

            if landmarks is not None:
                landmark5 = landmarks[face_index].astype(np.int)

                for landmark in range(landmark5.shape[0]):
                    color = landmark_to_color[landmark]
                    landmark_point = (landmark5[landmark][0], landmark5[landmark][1])
                    cv2.circle(img, landmark_point, CIRCLE_RADIUS, color, LINE_THICKNESS)

        print('Writing result image to: %s' % output_file_name)
        cv2.imwrite(output_file_name, img)


if __name__ == '__main__':
    img = read_image_file('worlds-largest-selfie.jpg')

    target_size = default_scale[0]
    max_size = default_scale[1]
    final_scale = calculate_correct_image_scale(img, target_size, max_size)

    faces, landmarks = detect_face(img, final_scale)
    output_file_name = './detector_test.jpg'
    draw_faces_and_landmarks(faces, landmarks, output_file_name)
