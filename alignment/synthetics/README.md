# Introduction

We provide training and testing tools on synthetics data.


## Dataset

### Training dataset

Download `Face Synthetics dataset` from [https://github.com/microsoft/FaceSynthetics](https://github.com/microsoft/FaceSynthetics) and put it somewhere.

<div align="left">
  <img src="https://github.com/microsoft/FaceSynthetics/raw/main/docs/img/dataset_samples_2.jpg" width="640"/>
</div>
<br/>

Then use [tools/prepare_synthetics.py](tools/prepare_synthetics.py) for training data preparation.


### Testing dataset

[300-W](https://ibug.doc.ic.ac.uk/resources/300-W/)


## Pretrained Model

[ResNet50d](https://drive.google.com/file/d/1kNP7qEl3AYNbaHFUg_ZiyRB1CtfDWXR4/view?usp=sharing)


## Train and Test

### Prerequisites

- pytorch_lightning
- timm
- albumentations

### Training

`` python -u trainer_synthetics.py ``

which uses `resnet50d` as backbone by default, please check the [code](trainer_synthetics.py) for detail.

### Testing

Please check [test_synthetics.py](test_synthetics.py) for detail.


## Result Visualization(3D 68 Keypoints)

<div align="left">
  <img src="https://github.com/nttstar/insightface-resources/blob/master/alignment/images/image_008_1.jpg?raw=true" width="320"/>
</div>

<div align="left">
  <img src="https://github.com/nttstar/insightface-resources/blob/master/alignment/images/image_017_1.jpg?raw=true" width="320"/>
</div>

<div align="left">
  <img src="https://github.com/nttstar/insightface-resources/blob/master/alignment/images/image_039.jpg?raw=true" width="320"/>
</div>


