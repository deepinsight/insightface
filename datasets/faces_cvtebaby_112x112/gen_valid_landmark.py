# execute this script in /root/MTCNN/mtcnn-pytorch/ cloned from https://github.com/TropComplique/mtcnn-pytorch.git
import sys
import os
from src import detect_faces, show_bboxes
from PIL import Image
import numpy as np
import shutil

data_root = "/data/victor/cvte_baby/valid/CVTE_baby_valid"
baby_landmark = open("/data/victor/cvte_baby/valid/landmark.txt", "w")
undetected_folder = '/data/victor/cvte_baby/valid/undetected'

label = 0
for dirs in os.listdir(data_root):
    for fn in os.listdir(os.path.join(data_root, dirs)):
        if not fn.endswith('.jpg'):
            continue
        image_path = os.path.join(data_root, dirs, fn)
        # if image_path != '/data/victor/cvte_baby/valid/CVTE_baby_valid/25/133_81612.jpg':
        #     continue
        img = Image.open(image_path)

        bounding_boxes, landmarks = detect_faces(img)
        bounding_boxes = np.array(bounding_boxes)

        areas = []
        for i in range(bounding_boxes.shape[0]):
            areas.append((bounding_boxes[i, 2] - bounding_boxes[i, 0]) * (bounding_boxes[i, 3] - bounding_boxes[i, 1]))

        if len(areas) == 0:
            print(image_path)
            shutil.copy(image_path, undetected_folder)
            if fn == '133_81612.jpg':
                d = [79, 73, 180, 180]
                p = np.array([(107,96), (163,93), (154,126), (124,153), (164,150)]).T.flatten()
            else:
                continue
        else:
            areas = np.array(areas)
            face_index = np.argmax(areas)
            d = bounding_boxes[face_index]
            p = landmarks[face_index]

        """
        tmp = show_bboxes(img, d, p)
        tmp.save("tmp.jpg")
        """

        baby_landmark.write('0\t')
        baby_landmark.write(image_path)
        baby_landmark.write('\t{}'.format(label))
        for i in range(4):
            baby_landmark.write("\t" + str(int(round(d[i]))))
        for i in range(5):
            baby_landmark.write("\t" + str(int(round(p[i]))) + "\t" + str(int(round(p[i+5]))))
        baby_landmark.write("\n")

    label += 1
