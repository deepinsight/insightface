### Introduction

 Here we provide some lightweight facial landmark models with fast coordinate regression.
 The input of these models is loose cropped face image while the output is the direct landmark coordinates. 


### Pretrained Models

- **Model ``2d106det``**

  Given face detection bounding box, predict 2d-106 landmarks. Mainly used for static image inference. Please check ``image_infer.py`` for detail.
  
  Download link: [baiducloud](https://pan.baidu.com/s/10m5GmtNV5snynDrq3KqIdg) (code: ``lqvv``)

- **Model ``2d106track``** 

  Given landmarks bounding box, predict 2d-106 landmarks. Used for video landmarks tracking.
  
  Download link: coming soon

### Visualization

![demoimg1](https://github.com/nttstar/insightface-resources/blob/master/alignment/images/t1_out.jpg)

### FAQ

