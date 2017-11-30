# InsightFace
Face Recognition Project

### Experiments

Default image size is 112x96 if not specified, all face images are aligned.

In ResNet setting, _v1 means original residual units.  _v2 means pre-activation units.  _v3 means BCBACB residual units.

_bo means using bottleneck residual units.

- **Softmax on LFW**

|   Network/Dataset   | VGG2@112x112 | WebFace | MS1M |  -   |  -   |
| :-----------------: | :----------: | :-----: | :--: | :--: | :--: |
|   SE-ResNet50_v3    |    0.9973    |    -    |  -   |      |      |
| SE-ResNet50\_v3\_bo |              |         |      |      |      |
|   SE-ResNet50_v1    |              |         |      |      |      |
|     ResNet50_v3     |      -       |    -    |  -   |      |      |
|  Inception-ResNet   |      -       |    -    |  -   |      |      |
| SE-Inception-ResNet |      -       |    -    |  -   |      |      |
|      MobileNet      |      -       |    -    |  -   |      |      |
|       ResNeXt       |      -       |    -    |  -   |      |      |
|                     |              |         |      |      |      |
