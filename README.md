# InsightFace
Face Recognition Project



### How to use

1. Download pre-aligned training dataset from our data repo which is a large binary file in MXnet .rec format(maybe ready soon), or align your dataset by yourself and then pack them to prevent random small files accessing. Check those scripts under src/common and src/align.
2. Run src/train_softmax.py to train your model and set proper parameters. For example, loss-type=0 means pure softmax while loss-type=1 means SphereLoss. It will output LFW accuracy every 2000 batches and save the model if necessary.

### Notes

Default image size is 112x96 if not specified, all face images are aligned.

In ResNet setting, \_v1 means original residual units.  \_v2 means pre-activation units.  \_v3 means BCBACB residual units.  LResNet means we use conv33+stride11 in its first convoluition layer instead of common conv77+stride22 to preserve high image resolution.   For ResNet50, we do not use bottleneck layers. For ResNet101 or ResNeXt101, we use.  

In last several layers, some different options can be tried to determine how embedding layer looks like and it may affect the performance. The whole network architecture can be thought as {ConvLayers(->GlobalPool)->EmbeddingLayer->Softmax}. Embedding size is set to 512 expect for optionA, as embedding size in optionA is determined by the filter size of last convolution group.

- OptionXD: Same with OptionD but use dropout after GP.  OptionAD is the default setting for inception series networks.
- OptionA: Use global pooling layer(GP). This is the default setting for all networks except inceptions.
- OptionB: Use one FC layer after GP.
- OptionC: Use FC->BN after GP.
- OptionD: Use FC->BN->PRelu after GP.
- OptionE: Use BN->Dropout->FC->BN after last conv layer.
- OptionF: Use BN->PRelu->Dropout->FC->BN after last conv layer.




### Experiments




- **Softmax only on VGG2@112x112**

|    Network/Dataset     |       LFW        |      ------      |      ------      |
| :--------------------: | :--------------: | :--------------: | :--------------: |
|      ResNet50D_v1      | 0.99350+-0.00293 |                  |                  |
|    SE-ResNet50A\_v1    | 0.99367+-0.00233 |                  |                  |
|    SE-ResNet50B_v1     | 0.99200+-0.00407 |                  |                  |
|    SE-ResNet50C_v1     | 0.99317+-0.00404 |                  |                  |
|    SE-ResNet50D_v1     | 0.99383+-0.00259 |                  |                  |
|    SE-ResNet50E\_v1    | 0.99267+-0.00343 |                  |                  |
|    SE-ResNet50F\_v1    | 0.99367+-0.00194 |                  |                  |
|    SE-LResNet50C_v1    | 0.99567+-0.00238 |                  |                  |
|    SE-LResNet50D_v1    | 0.99600+-0.00281 |                  |                  |
|    SE-LResNet50E_v1    | 0.99650+-0.00174 |        -         |        -         |
|    SE-LResNet50A_v3    | 0.99583+-0.00327 |                  |                  |
|    SE-LResNet50D_v3    | 0.99617+-0.00358 |        -         |        -         |
|    SE-LResNet50E_v3    | 0.99767+-0.00200 |        -         |        -         |
|     LResNet50E_v3      |                  |                  |                  |
|    SE-LResNet50F_v3    |                  |                  |                  |
|    SE-LResNet50G_v3    | 0.99350+-0.00263 |                  |                  |
|    SE-ResNet101D_v3    | 0.99517+-0.00252 |                  |                  |
|    SE-ResNet101E_v3    | 0.99467+-0.00221 |                  |                  |
|    SE-ResNet152E_v3    | 0.99500+-0.00307 |                  |                  |
|   Inception-ResNetAD   | 0.99417+-0.00375 |        -         |        -         |
| SE-Inception-ResNet-v2 |        -         |        -         |        -         |
|       MobileNetD       | 0.99150+-0.00425 |        -         |        -         |
|      LMobileNetD       |                  |                  |                  |
|      LMobileNetE       | 0.99600+-0.00281 |        -         |        -         |
|      LMobileNetF       |                  |                  |                  |
|    LResNeXt101E_v3     |                  |                  |                  |
