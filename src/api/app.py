from flask import Flask
import face_model
import argparse
import json
import base64
#import requests
import numpy as np
import urllib
import cv2
from flask import Flask, render_template, request, jsonify


parser = argparse.ArgumentParser(description='do verification')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../model/softmax,50', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)

app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello, This is InsightFace!'

def image_resize(image):
  m = min(image.shape[0], image.shape[1])
  f = 640.0/m
  if f<1.0:
    image = cv2.resize(image, (int(image.shape[1]*f), int(image.shape[0]*f)))
  return image

def get_image(data):
  image = None
  if 'url' in data:
    url = data['url']
    if url.startswith('http'):
      resp = urllib.urlopen(url)
      image = np.asarray(bytearray(resp.read()), dtype="uint8")
      image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
      image = cv2.imread(url, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image_resize(image)
  elif 'data' in data:
    _bin = data['data']
    if _bin is not None:
      if not isinstance(_bin, list):
        _bin = base64.b64decode(_bin)
        _bin = np.fromstring(_bin, np.uint8)
        image = cv2.imdecode(_bin, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_resize(image)
      else:
        image = []
        for __bin in _bin:
          __bin = base64.b64decode(__bin)
          __bin = np.fromstring(__bin, np.uint8)
          _image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
          _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
          _image = image_resize(_image)
          image.append(_image)

  return image

@app.route('/ver', methods=['POST'])
def ver():
  try:
    data = request.data
    values = json.loads(data)
    source_image = get_image(values['source'])
    if source_image is None:
      print('source image is None')
      return '-1'
    assert not isinstance(source_image, list)
    print(source_image.shape)
    target_image = get_image(values['target'])
    if target_image is None:
      print('target image is None')
      return '-1'
    #print(target_image.shape)
    if not isinstance(target_image, list):
      target_image = [target_image]
    #print('before call')
    #ret = model.is_same_id(source_image, target_image)
    ret = model.sim(source_image, target_image)
  except Exception as ex:
    print(ex)
    return '-1'

  #return str(int(ret))
  print('sim', ret)
  return "%1.3f"%ret

if __name__ == '__main__':
    app.run('0.0.0.0', port=18080, debug=False)
