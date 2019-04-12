import face_model
import argparse
import cv2
import sys
import numpy as np
import datetime
import face_preprocess
import mxnet as mx


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='model/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id') # pass -1 to run with CPU
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
args = parser.parse_args()

model = face_model.FaceModel(args)
#img = cv2.imread('Tom_Hanks_54745.png')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set properties. Each returns === True on success (i.e. correct resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while(True):
	# Capture frame-by-frame
        
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        results = model.get_bbox_and_landmarks(frame)

        if results is None:
            continue
        else:
            bboxes, points = results
        
        #number of detected faces in the frame
        num_faces = bboxes.shape[0]
        for face in range(num_faces):
            bbox = bboxes[face,0:4]
            conf = bboxes[face,-1]
            xmin=int(bbox[0])
            ymin=int(bbox[1])
            xmax=int(bbox[2])
            ymax=int(bbox[3])
            
            try:
                point = points[face,:].reshape((2,5)).T
                nimg = face_preprocess.preprocess(frame, bbox, point, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                aligned = np.transpose(nimg, (2,0,1))
                input_blob = np.expand_dims(aligned, axis=0)
                data = mx.nd.array(input_blob)
                img = mx.io.DataBatch(data=(data,))
                gender, age = model.get_ga(img)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0),3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                if gender==0:
                    sex='Female'
                elif gender==1:
                    sex='Male'
                    text = sex +' '+ str(age)
                    cv2.putText(frame, text,(xmin, ymax-ymin), font, 1,(255,255,255),2,cv2.LINE_AA)
            except ValueError:
                pass
            
        # Display the resulting frame
        cv2.imshow('wecam',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()












#f1 = model.get_feature(img)
#print(f1[0:10])
# for _ in range(5):
#   gender, age = model.get_ga(img)
# time_now = datetime.datetime.now()
# count = 200
# for _ in range(count):
#   gender, age = model.get_ga(img)
# time_now2 = datetime.datetime.now()
# diff = time_now2 - time_now
# print('time cost', diff.total_seconds()/count)
# print('gender is',gender)
# print('age is', age)


    # bbox = bbox[0,0:4]
    # points = points[0,:].reshape((2,5)).T
    # print(bbox)
    # print(points)
    # nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    # aligned = np.transpose(nimg, (2,0,1))
    # input_blob = np.expand_dims(aligned, axis=0)
    # data = mx.nd.array(input_blob)
    # db = mx.io.DataBatch(data=(data,))

