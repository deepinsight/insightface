### Introduction

 Here we provide some lightweight facial landmark models with fast coordinate regression.
 The input of these models is loose cropped face image while the output is the direct landmark coordinates. 


### Pretrained Models

- **Model ``2d106det``**

  **2021.07: We now support model inference by our `insightface` python package, please check [image_infer.py](image_infer.py) for detail.**

  Given face detection bounding box, predict 2d-106 landmarks. Mainly used for static image inference. 
  
  Backbone: MobileNet-0.5, size 5MB.
  
  Input: size 192x192, loose cropped detection bounding-box.
  
  Download link:
  
  [baidu cloud](https://pan.baidu.com/s/10m5GmtNV5snynDrq3KqIdg) (code: ``lqvv``)
  
  [google drive](https://drive.google.com/file/d/13Pz8mH-a1s7RXpq_jFUXxaqCpDUE0oSr/view?usp=sharing)
  
  

- **Model ``2d106track``** 

  Given landmarks bounding box, predict 2d-106 landmarks. Used for video landmarks tracking.
  
  Download link: coming soon

### Visualization


<p align="center">Points mark-up(ordered by point names):</p>

<div align="center">
	<img src="https://github.com/nttstar/insightface-resources/blob/master/alignment/images/2d106markup.jpg" alt="markup" width="320">
</div>


<p align="center">Image result:</p>

<div align="center">
	<img src="https://github.com/nttstar/insightface-resources/blob/master/alignment/images/t1_out.jpg" alt="imagevis" width="800">
</div>


<p align="center">Video result:</p>

<div align="center">
	<img src="https://github.com/nttstar/insightface-resources/blob/master/alignment/images/C_jiaguo.gif" alt="videovis" width="240">
</div>


### FAQ

