# InsightFace Model Zoo

:bell:   **ALL models are available for non-commercial research purposes only.**

## 0. Python Package models

To check the detail of insightface python package, please see [here](../python-package).

To install: ``pip install -U insightface``

To use the specific model pack:

```
model_pack_name = 'buffalo_l'
app = FaceAnalysis(name=model_pack_name)
```

Name in **bold** is the default model pack in latest version.


| Name           | Detection Model | Recognition Model   | Alignment    | Attributes | Model-Size |
| -------------- | --------------- | ------------------- | ------------ | ---------- | ---------- |
| antelopev2 | RetinaFace-10GF      | ResNet100@Glint360K | 2d106 & 3d68 | Gender&Age | 407MB |
| **buffalo_l**      | RetinaFace-10GF      | ResNet50@WebFace600K | 2d106 & 3d68 | Gender&Age | 326MB |
| buffalo_m      | RetinaFace-2.5GF     | ResNet50@WebFace600K | 2d106 & 3d68 | Gender&Age | 313MB |
| buffalo_s      | RetinaFace-500MF     | MBF@WebFace600K | 2d106 & 3d68 | Gender&Age | 159MB |
| buffalo_sc      | RetinaFace-500MF     | MBF@WebFace600K | - | - | 16MB |

### Recognition accuracy of python library model packs:

| Name      | MR-ALL | African | Caucasian | South Asian | East Asian | LFW    | CFP-FP | AgeDB-30 | IJB-C(E4) |
| :-------- | ------ | ------- | --------- | ----------- | ---------- | ------ | ------ | -------- | --------- |
| buffalo_l | 91.25  | 90.29   | 94.70     | 93.16       | 74.96      | 99.83  | 99.33  | 98.23    | 97.25     |
| buffalo_s	      | 71.87 | 69.45  | 80.45    | 73.39      | 51.03     | 99.70 | 98.00  | 96.58    | 95.02 |

*buffalo_m has the same accuracy with buffalo_l.*

*buffalo_sc has the same accuracy with buffalo_s.*

(Note that almost all ONNX models in our model_zoo can be called by python library.)

##  1. Face Recognition models.

### Definition:

The default training loss is margin based softmax if not specified.

``MFN``: MobileFaceNet

``MS1MV2``: MS1M-ArcFace

``MS1MV3``: MS1M-RetinaFace

``MS1M_MegaFace``: MS1MV2+MegaFace_train

``_pfc``: using Partial FC, with sample-ratio=0.1

``MegaFace``: MegaFace identification test, with gallery=1e6.

``IJBC``: IJBC 1:1 test, under FAR<=1e-4.

``BDrive``: BaiduDrive

``GDrive``: GoogleDrive

### List of models by MXNet and PaddlePaddle:

| Backbone | Dataset | Method  | LFW   | CFP-FP | AgeDB-30 | MegaFace | Link.                                                        |
| -------- | ------- | ------- | ----- | ------ | -------- | -------- | ------------------------------------------------------------ |
| R100 (mxnet)     | MS1MV2  | ArcFace | 99.77 | 98.27  | 98.28    | 98.47    | [BDrive](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA), [GDrive](https://drive.google.com/file/d/1Hc5zUfBATaXUgcU2haUNa7dcaZSw95h2/view?usp=sharing) |
| MFN (mxnet)     | MS1MV1  | ArcFace | 99.50 | 88.94  | 95.91    | -        | [BDrive](https://pan.baidu.com/s/1If28BkHde4fiuweJrbicVA), [GDrive](https://drive.google.com/file/d/1RHyJIeYuHduVDDBTn3ffpYEZoXWRamWI/view?usp=sharing) |
| MFN (paddle)     | MS1MV2  | ArcFace | 99.45 | 93.43  | 96.13    |  -   | [pretrained model](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/MobileFaceNet_128_v1.0_pretrained.tar), [inference model](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/mobileface_v1.0_infer.tar) |
| iResNet50 (paddle)     | MS1MV2  | ArcFace | 99.73 | 97.43  | 97.88    |  -   | [pretrained model](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/arcface_iresnet50_v1.0_pretrained.tar), [inference model](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/arcface_iresnet50_v1.0_infer.tar) |



### List of models by various depth IResNet and training datasets:

| Backbone | Dataset   | MR-ALL | African | Caucasian | South Asian | East Asian | Link(onnx)                                                            |
|----------|-----------|--------|---------|-----------|-------------|------------|-----------------------------------------------------------------------|
| R100     | Casia     | 42.735 | 39.666  | 53.933    | 47.807      | 21.572     | [GDrive](https://drive.google.com/file/d/1WOrOK-qZO5FcagscCI3td6nnABUPPepD/view?usp=sharing) |
| R100     | MS1MV2    | 80.725 | 79.117  | 87.176    | 85.501      | 55.807     | [GDrive](https://drive.google.com/file/d/1772DTho9EG047KNUIv2lop2e7EobiCFn/view?usp=sharing) |
| R18      | MS1MV3    | 68.326 | 62.613  | 75.125    | 70.213      | 43.859     | [GDrive](https://drive.google.com/file/d/1dWZb0SLcdzr-toUzsVZ1zogn9dEIW1Dk/view?usp=sharing) |
| R34      | MS1MV3    | 77.365 | 71.644  | 83.291    | 80.084      | 53.712     | [GDrive](https://drive.google.com/file/d/1ON6ImX-AigDKAi4pelFPf12vkJVyGFKl/view?usp=sharing) |
| R50      | MS1MV3    | 80.533 | 75.488  | 86.115    | 84.305      | 57.352     | [GDrive](https://drive.google.com/file/d/1FPldzmZ6jHfaC-R-jLkxvQRP-cLgxjCT/view?usp=sharing) |
| R100     | MS1MV3    | 84.312 | 81.083  | 89.040    | 88.082      | 62.193     | [GDrive](https://drive.google.com/file/d/1fZOfvfnavFYjzfFoKTh5j1YDcS8KCnio/view?usp=sharing) |
| R18      | Glint360K | 72.074 | 68.230  | 80.575    | 75.852      | 47.831     | [GDrive](https://drive.google.com/file/d/1Z0eoO1Wqv32K8TdFHKqrlrxv46_W4390/view?usp=sharing) |
| R34      | Glint360K | 83.015 | 79.907  | 88.620    | 86.815      | 60.604     | [GDrive](https://drive.google.com/file/d/1G1oeLkp_b3JA_z4wGs62RdLpg-u_Ov2Y/view?usp=sharing) |
| R50      | Glint360K | 87.077 | 85.272  | 91.617    | 90.541      | 66.813     | [GDrive](https://drive.google.com/file/d/1MpRhM76OQ6cTzpr2ZSpHp2_CP19Er4PI/view?usp=sharing) |
| R100     | Glint360K | 90.659 | 89.488  | 94.285    | 93.434      | 72.528     | [GDrive](https://drive.google.com/file/d/1Gh8C-bwl2B90RDrvKJkXafvZC3q4_H_z/view?usp=sharing) |


### List of models by IResNet-50 and different training datasets:

| Dataset           | MR-ALL | African | Caucasian | South Asian | East Asian | LFW   | CFP-FP | AgeDB-30 | IJB-C(E4) | Link(onnx) |
| :--------         | ------ | ------- | ----      | ------      | --------   | ----- | ------ | -------- | --------- | --- |
| CISIA	            | 36.794 | 42.550  | 55.825    | 49.618      | 19.611     | 99.450| 95.214 | 94.900   | 87.220    | [GDrive](https://drive.google.com/file/d/1km-cVFvUAPU1UumLLi1fIRasdg6VA-vM/view?usp=sharing) |
| CISIA_pfc	        | 37.107 | 38.934  | 53.823    | 48.674      | 19.927     | 99.367| 95.429 | 94.600   | 84.970    | [GDrive](https://drive.google.com/file/d/1z8linstTZopL5Yy7NOUgVVtgzGtsu1LM/view?usp=sharing) |
| VGG2	            | 38.578 | 35.259  | 54.304    | 44.081      | 24.095     | 99.550| 97.410 | 95.080   | 91.220    | [GDrive](https://drive.google.com/file/d/1UwyVIDSNDkHKClBANrWi8qpMU4nXizT6/view?usp=sharing) |
| VGG2_pfc	        | 40.673 | 36.767  | 60.180    | 49.039      | 24.255     | 99.683| 98.529 | 95.400   | 92.490    | [GDrive](https://drive.google.com/file/d/1uW0EsctVyPklSyXMXF39AniIhSRXCRtp/view?usp=sharing) |
| GlintAsia	        | 62.663 | 49.531  | 64.829    | 57.984      | 61.743     | 99.583| 93.186 | 95.400   | 91.500    | [GDrive](https://drive.google.com/file/d/1IyXh7m1HMwTZw4B5N1WMPIsN-S9kdS95/view?usp=sharing) |
| GlintAsia_pfc	    | 63.149 | 50.366  | 65.227    | 57.936      | 61.820     | 99.650| 93.029 | 95.233   | 91.140    | [GDrive](https://drive.google.com/file/d/1CTjalggNucgPkmpFi5ij-NGG1Fy9sL5r/view?usp=sharing) |
| MS1MV2	        | 77.696 | 74.596  | 84.126    | 82.041      | 51.105     | 99.833| 98.083 | 98.083   | 96.140    | [GDrive](https://drive.google.com/file/d/1rd4kbiXtXBTWE8nP7p4OTv_CAp2FUa1i/view?usp=sharing) |
| MS1MV2_pfc	    | 77.738 | 74.728  | 84.883    | 82.798      | 52.507     | 99.783| 98.071 | 98.017   | 96.080    | [GDrive](https://drive.google.com/file/d/1ryrXenGQa-EGyk64mVaG136ihNUBmNMW/view?usp=sharing) |
| MS1M_MegaFace	    | 78.372 | 74.138  | 82.251    | 77.223      | 60.203     | 99.750| 97.557 | 97.400   | 95.350    | [GDrive](https://drive.google.com/file/d/1c2JG0StcTMDrL4ywz3qWTN_9io3lo_ER/view?usp=sharing) |
| MS1M_MegaFace_pfc | 78.773 | 73.690  | 82.947    | 78.793      | 57.566     | 99.800| 97.870 | 97.733   | 95.400    | [GDrive](https://drive.google.com/file/d/1BnG48LS_HIvYlSbSnP_LzpO3xjx0_rpu/view?usp=sharing) |
| MS1MV3	        | 82.522 | 77.172  | 87.028    | 86.006      | 60.625     | 99.800| 98.529 | 98.267   | 96.580    | [GDrive](https://drive.google.com/file/d/1Tqorubgcl0qfjbjEM_Y9EDmjG5tCWzbr/view?usp=sharing) |
| MS1MV3_pfc	    | 81.683 | 78.126  | 87.286    | 85.542      | 58.925     | 99.800| 98.443 | 98.167   | 96.430    | [GDrive](https://drive.google.com/file/d/15jrHCqhEmoSZ93kKL9orVMhbKfNWAhp-/view?usp=sharing) |
| Glint360k	        | 86.789 | 84.749  | 91.414    | 90.088      | 66.168     | 99.817| 99.143 | 98.450   | 97.130    | [GDrive](https://drive.google.com/file/d/1gnt6P3jaiwfevV4hreWHPu0Mive5VRyP/view?usp=sharing) |
| Glint360k_pfc	    | 87.077 | 85.272  | 91.616    | 90.541      | 66.813     | 99.817| 99.143 | 98.450   | 97.020    | [GDrive](https://drive.google.com/file/d/164o2Ct42tyJdQjckeMJH2-7KTXolu-EP/view?usp=sharing) |
| WebFace600K	    | 90.566 | 89.355  | 94.177    | 92.358      | 73.852     | 99.800| 99.200 | 98.100   | 97.120    | [GDrive](https://drive.google.com/file/d/1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg/view?usp=sharing) |
| WebFace600K_pfc    | 89.951 | 89.301  | 94.016    | 92.381      | 73.007     | 99.817| 99.143 | 98.117   | 97.010    | [GDrive](https://drive.google.com/file/d/11TASXssTnwLY1ZqKlRjsJiV-1nWu9pDY/view?usp=sharing) |
| Average	        | 69.247 | 65.908  | 77.121    | 72.819      | 52.014     | 99.706| 97.374 | 96.962   | 93.925    |  |
| Average_pfc	    | 69.519 | 65.898  | 77.497    | 73.213      | 51.853     | 99.715| 97.457 | 96.965   | 93.818    |  |

### List of models by MobileFaceNet and different training datasets:

**``FLOPS``:** 450M FLOPs

**``Model-Size``:** 13MB

| Dataset           | MR-ALL | African | Caucasian | South Asian | East Asian | LFW   | CFP-FP | AgeDB-30 | IJB-C(E4) | Link(onnx) |
| :--------         | ------ | ------- | ----      | ------      | --------   | ----- | ------ | -------- | --------- | --- |
| WebFace600K	      | 71.865 | 69.449  | 80.454    | 73.394      | 51.026     | 99.70 | 98.00  | 96.58    | 95.02     | - |


## 2. Face Detection models.

### 2.1 RetinaFace

In RetinaFace, mAP was evaluated with multi-scale testing.

``m025``: means MobileNet-0.25

| Impelmentation           | Easy-Set | Medium-Set | Hard-Set | Link                                                         |
| ------------------------ | -------- | ---------- | -------- | ------------------------------------------------------------ |
| RetinaFace-R50           | 96.5     | 95.6       | 90.4     | [BDrive](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ), [GDrive](https://drive.google.com/file/d/1wm-6K688HQEx_H90UdAIuKv-NAsKBu85/view?usp=sharing) |
| RetinaFace-m025(yangfly) | -        | -          | 82.5     | [BDrive](https://pan.baidu.com/s/1P1ypO7VYUbNAezdvLm2m9w)(nzof), [GDrive](https://drive.google.com/drive/folders/1OTXuAUdkLVaf78iz63D1uqGLZi4LbPeL?usp=sharing) |
| BlazeFace-FPN-SSH	(paddle)           | 91.9     | 89.8       | 81.7%     | [pretrained model](https://paddledet.bj.bcebos.com/models/blazeface_fpn_ssh_1000e.pdparams), [inference model](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/blazeface_fpn_ssh_1000e_v1.0_infer.tar) |

### 2.2 SCRFD

In SCRFD, mAP was evaluated with single scale testing, VGA resolution.

``2.5G``: means the model cost ``2.5G`` FLOPs while the input image is in VGA(640x480) resolution.

``_KPS``: means this model can detect five facial keypoints.

|      Name      | Easy  | Medium | Hard  | FLOPs | Params(M) | Infer(ms) | Link(pth)                                                    |
| :------------: | ----- | ------ | ----- | ----- | --------- | --------- | ------------------------------------------------------------ |
|   SCRFD_500M   | 90.57 | 88.12  | 68.51 | 500M  | 0.57      | 3.6       | [GDrive](https://drive.google.com/file/d/1OX0i_vWDp1Fp-ZynOUMZo-q1vB5g1pTN/view?usp=sharing) |
|    SCRFD_1G    | 92.38 | 90.57  | 74.80 | 1G    | 0.64      | 4.1       | [GDrive](https://drive.google.com/file/d/1acd5wKjWnl1zMgS5YJBtCh13aWtw9dej/view?usp=sharing) |
|   SCRFD_2.5G   | 93.78 | 92.16  | 77.87 | 2.5G  | 0.67      | 4.2       | [GDrive](https://drive.google.com/file/d/1wgg8GY2vyP3uUTaAKT0_MSpAPIhmDsCQ/view?usp=sharing) |
|   SCRFD_10G    | 95.16 | 93.87  | 83.05 | 10G   | 3.86      | 4.9       | [GDrive](https://drive.google.com/file/d/1kUYa0s1XxLW37ZFRGeIfKNr9L_4ScpOg/view?usp=sharing) |
|   SCRFD_34G    | 96.06 | 94.92  | 85.29 | 34G   | 9.80      | 11.7      | [GDrive](https://drive.google.com/file/d/1w9QOPilC9EhU0JgiVJoX0PLvfNSlm1XE/view?usp=sharing) |
| SCRFD_500M_KPS | 90.97 | 88.44  | 69.49 | 500M  | 0.57      | 3.6       | [GDrive](https://drive.google.com/file/d/1TXvKmfLTTxtk7tMd2fEf-iWtAljlWDud/view?usp=sharing) |
| SCRFD_2.5G_KPS | 93.80 | 92.02  | 77.13 | 2.5G  | 0.82      | 4.3       | [GDrive](https://drive.google.com/file/d/1KtOB9TocdPG9sk_S_-1QVG21y7OoLIIf/view?usp=sharing) |
| SCRFD_10G_KPS  | 95.40 | 94.01  | 82.80 | 10G   | 4.23      | 5.0       | [GDrive](https://drive.google.com/file/d/1-2uy0tgkenw6ZLxfKV1qVhmkb5Ep_5yx/view?usp=sharing) |



## 3. Face Alignment models.

### 2.1 2D Face Alignment

| Impelmentation        | Points | Backbone      | Params(M) | Link(onnx)                                                   |
| --------------------- | ------ | ------------- | --------- | ------------------------------------------------------------ |
| Coordinate-regression | 106    | MobileNet-0.5 | 1.2       | [GDrive](https://drive.google.com/file/d/1M5685m-bKnMCt0u2myJoEK5gUY3TDt_1/view?usp=sharing) |

### 2.2 3D Face Alignment

| Impelmentation | Points | Backbone  | Params(M) | Link(onnx)                                                   |
| -------------- | ------ | --------- | --------- | ------------------------------------------------------------ |
| -              | 68     | ResNet-50 | 34.2      | [GDrive](https://drive.google.com/file/d/1aJe5Rzoqrtf_a9U84E-V1b0rUi8-QbCI/view?usp=sharing) |

### 2.3 Dense Face Alignment

## 4. Face Attribute models.

### 4.1 Gender&Age 

| Training-Set | Backbone       | Params(M) | Link(onnx)                                                   |
| ------------ | -------------- | --------- | ------------------------------------------------------------ |
| CelebA       | MobileNet-0.25 | 0.3       | [GDrive](https://drive.google.com/file/d/1Mm3TeUuaZOwmEMp0nGOddvgXCjpRodPU/view?usp=sharing) |


### 4.2 Expression
