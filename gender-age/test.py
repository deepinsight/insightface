import face_model
import argparse
import cv2
import sys
import numpy as np
import datetime

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--image', default='Tom_Hanks_54745.png', help='')
parser.add_argument('--model',
                    default='model/model,0',
                    help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument(
    '--det',
    default=0,
    type=int,
    help='mtcnn option, 1 means using R+O, 0 means detect from begining')
args = parser.parse_args()

model = face_model.FaceModel(args)
#img = cv2.imread('Tom_Hanks_54745.png')
img = cv2.imread(args.image)
img = model.get_input(img)
#f1 = model.get_feature(img)
#print(f1[0:10])
for _ in range(5):
    gender, age = model.get_ga(img)
time_now = datetime.datetime.now()
count = 200
for _ in range(count):
    gender, age = model.get_ga(img)
time_now2 = datetime.datetime.now()
diff = time_now2 - time_now
print('time cost', diff.total_seconds() / count)
print('gender is', gender)
print('age is', age)
