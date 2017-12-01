# InsightFace
Face Recognition Project

### Experiments

Default image size is 112x96 if not specified, all face images are aligned.

In ResNet setting, _v1 means original residual units.  _v2 means pre-activation units.  _v3 means BCBACB residual units.  LResNet means we use conv33+stride11 in its first convoluition layer instead of common conv77+stride22 to preserve high image resolution._  _bo means using bottleneck residual units.   

In last several layers, some different options can be tried to determine how embedding layer looks like and it may affect the performance. The whole network architecture can be thought as (ConvLayers[->GlobalPool]->EmbeddingLayer->Softmax). Embedding size is set to 512 expect for optionA, as embedding size in optionA is determined by the filter size of last convolution group.

- OptionA: Use the final global pooling layer(GP) output as embedding layer directly.
- OptionB: Use one FC layer after GP.
- OptionC: Use FC->BN after GP.
- OptionD: Use FC->BN->PRelu after GP.
- OptionE: Use Dropout->FC->BN after last conv layer.

'NB' means we do not insert one BN layer after embedding layer(FC_embedding->FC->softmax instead of FC_embedding->BN->FC->softmax).

- **Softmax on LFW**

|   Network/Dataset   |   VGG2@112x112   | WebFace | MS1M |  -   |  -   |
| :-----------------: | :--------------: | :-----: | :--: | :--: | :--: |
|  SE-LResNet50E_v3   |      0.9973      |    -    |  -   |      |      |
|   SE-ResNet50C_v1   | 0.99217+-0.00236 |         |      |      |      |
|   SE-ResNet50B_v1   |  Not Converged   |         |      |      |      |
|   SE-ResNet50D_v1   |                  |         |      |      |      |
|  SE-ResNet50A\_v1   |                  |         |      |      |      |
|  SE-LResNet50C_v1   |                  |         |      |      |      |
|  SE-LResNet50E_v1   |        -         |    -    |  -   |      |      |
|  Inception-ResNet   |        -         |    -    |  -   |      |      |
| SE-Inception-ResNet |        -         |    -    |  -   |      |      |
|      MobileNet      |        -         |    -    |  -   |      |      |
|       ResNeXt       |        -         |    -    |  -   |      |      |
|                     |                  |         |      |      |      |
