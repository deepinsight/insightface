import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis

parser = argparse.ArgumentParser(description='insightface test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
args = parser.parse_args()

app = FaceAnalysis(name='antelope')
app.prepare(ctx_id=args.ctx, det_size=(224,224))

img = cv2.imread('Tom_Hanks_54745.png')
faces = app.get(img)
print(len(faces))
for face in faces:
    print(face.bbox)
    print(face.kps)
    print(face.embedding.shape)

