import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis

assert insightface.__version__>='0.2'

parser = argparse.ArgumentParser(description='insightface test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
args = parser.parse_args()

app = FaceAnalysis(name='antelope')
app.prepare(ctx_id=args.ctx, det_size=(640,640))

img = cv2.imread('../sample-images/t1.jpg')
faces = app.get(img)
assert len(faces)==6
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
print(len(faces))
for face in faces:
    print(face.bbox)
    print(face.kps)
    print(face.embedding.shape)

