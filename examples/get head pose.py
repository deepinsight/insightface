import json
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis

ESTIMATE_EYE_DISTANCE = 6  # cm

app = FaceAnalysis(
    allowed_modules=["detection",
                     "landmark_3d_68",
                     ], providers=["CUDAExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

TEST_IMAGE_PATH = r'C:\Users\BinhNguyen\Downloads\test_images'
test_images_paths = glob(TEST_IMAGE_PATH + '/*.*')
test_images_paths = [path for path in test_images_paths if not path.endswith('.json')]

for image_path in tqdm(test_images_paths):
    mat = cv2.imread(image_path)
    g_output = json.load(open(image_path + '.json'))
    g_tile_angle = g_output['tilt_angle']
    g_pan_angle = g_output['pan_angle']
    g_roll_angle = g_output['roll_angle']
    print(mat.shape)

    faces = app.get(mat)

    for face in faces:
        landmark = face.landmark_3d_68

        # get head pose from landmarks
        pose = face.pose
        mean = np.mean(landmark, axis=0)
        left_eye = np.mean(landmark[42], axis=0)
        right_eye = np.mean(landmark[39], axis=0)
        eye_distance = np.linalg.norm(left_eye - right_eye)
        scaled = (landmark / eye_distance) * ESTIMATE_EYE_DISTANCE * 10 ** -2
        centered = scaled - np.mean(scaled, axis=0)

        rotation_matrix = np.empty((3, 3))
        rotation_matrix[0, :] = (centered[16] - centered[0]) / np.linalg.norm(centered[16] - centered[0])
        rotation_matrix[1, :] = (centered[27] - centered[8]) / np.linalg.norm(centered[27] - centered[8])
        rotation_matrix[2, :] = np.cross(rotation_matrix[0, :], rotation_matrix[1, :])
        inverted_rotation = np.linalg.inv(rotation_matrix)

        object_point = centered.dot(inverted_rotation)

        # draw head pose
        for i, (point, object_point) in enumerate(zip(landmark, object_point)):
            cv2.circle(mat, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            # draw the X axis
            cv2.line(mat, mean[:2].astype(int),
                     (mean + rotation_matrix[0, :] * 100)[:2].astype(int), (255, 0, 0), 2)
            # draw the Y axis
            cv2.line(mat, mean[:2].astype(int),
                     (mean + rotation_matrix[1, :] * 100)[:2].astype(int), (0, 255, 0), 2)
            # draw the Z axis
            cv2.line(mat, mean[:2].astype(int),
                     (mean + rotation_matrix[2, :] * 100)[:2].astype(int), (0, 0, 255), 2)

        # convert rotation matrix to euler angles

        sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])

        if sy > 1e-6:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        print("x: {:.2f}, y: {:.2f}, z: {:.2f}".format(x, y, z))

        angles = np.array([x, y, z])
        sin_angles = np.sin(angles)
        arc_sin_angles = np.arcsin(sin_angles)
        degrees = np.degrees(arc_sin_angles)
        pitch, yaw, roll = degrees
        yaw = -yaw

        pitch2, yaw2, roll2 = pose
        print("Solution 1: pitch: {:.2f}, yaw: {:.2f}, roll: {:.2f} in Deg".format(pitch, yaw, roll))
        print("Solution 2: pitch: {:.2f}, yaw: {:.2f}, roll: {:.2f} in Deg".format(pitch2, yaw2, roll2))
        print(f"Google API: pitch: {g_tile_angle}, yaw: {g_pan_angle}, roll: {g_roll_angle}")

        delta_1 = abs(pitch - float(g_tile_angle)) + abs(yaw - float(g_pan_angle)) + abs(roll - float(g_roll_angle))
        delta_2 = abs(pitch2 - float(g_tile_angle)) + abs(yaw2 - float(g_pan_angle)) + abs(roll2 - float(g_roll_angle))
        print(f"Delta 1: {delta_1}")
        print(f"Delta 2: {delta_2}")
