from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


app = FaceAnalysis(
    allowed_modules=["detection",
                     "landmark_3d_68",
                     ], providers=["CUDAExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

mat = ins_get_image('t1')

print(mat.shape)

faces = app.get(mat)

for face in faces:
    pitch, yaw, roll = face.pose
    print(f'pitch: {pitch:2f}, yaw: {yaw}, roll: {roll}')