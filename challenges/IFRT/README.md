# InsightFace Recognition Test (IFRT)
**IFRT** is a globalised fair benchmark for face recognition algorithms. IFRT evaluates the algorithm performance on worldwide web pictures which contain various sex, age and race groups, but no identification photos.

**IFRT** testset consists of non-celebrities so we can ensure that it has very few overlap with public available face recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. As the result, we can evaluate the FAIR performance for different algorithms.

Similar to [FRVT](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt), we encourage participants to prepare a black-box feature extractor or raw model files.

## Dataset Statistics and Visualization

IFRT testset contains 242,143 identities and 1,624,305 images.

| Race-Set     | Identities  | Images        |  Positive Pairs   | Negative Pairs        |
| -------      | ----------  | -----------   |  -----------      | -----------           |
| African      | 43,874      | 298,010       |  870,091          | 88,808,791,999        |
| Caucasian    | 103,293     | 697,245       |  2,024,609        | 486,147,868,171       |
| Indian       | 35,086      | 237,080       |  688,259          | 56,206,001,061        |
| Asian        | 59,890      | 391,970       |  1,106,078        | 153,638,982,852       |
| **ALL**      | **242,143** | **1,624,305** |  **4,689,037**    | **2,638,360,419,683** |

<details>
  <summary>Click to check the sample images(here we manually blur it to protect privacy) </summary>
  <img src="https://github.com/nttstar/insightface-resources/blob/master/images/ifrtsample_blur.jpg" alt="ifrtsample" width="640">
</details>

## Evaluation Metric

For ``Mask`` set, TAR is measured on mask-to-nonmask 1:1 protocal, with FAR less than 0.0001(e-4).

For ``Children`` set, TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.0001(e-4).

For other sets, TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6).

## Baselines

**``2021.04.25``** We made a clean on Asian subset.

| Backbone   | Dataset    | Method     | Mask   | Children | African | Caucasian | South Asian | East Asian | All    | size(mb) | infer(ms) |
|------------|------------|------------|--------|----------|---------|-----------|-------------|------------|--------|----------|-----------|
| R100  | Casia  | ArcFace  | 26.623 | 30.359   | 39.666  | 53.933    | 47.807      | 21.572     | 42.735 | 248.904  | 7.073     |
| R100  | MS1MV2  | ArcFace  | 65.767 | 60.496   | 79.117  | 87.176    | 85.501      | 55.807     | 80.725 | 248.904  | 7.028     |
| R18  | MS1MV3  | ArcFace | 47.853 | 41.047   | 62.613  | 75.125    | 70.213      | 43.859     | 68.326 | 91.658   | 1.856     |
| R34  | MS1MV3  | ArcFace | 58.723 | 55.834   | 71.644  | 83.291    | 80.084      | 53.712     | 77.365 | 130.245  | 3.054     |
| R50  | MS1MV3  | ArcFace | 63.850 | 60.457   | 75.488  | 86.115    | 84.305      | 57.352     | 80.533 | 166.305  | 4.262     |
| R100 | MS1MV3 | ArcFace | 69.091 | 66.864   | 81.083  | 89.040    | 88.082      | 62.193     | 84.312 | 248.590  | 7.031     |
| R18   | Glint360K   | ArcFace | 53.317 | 48.113   | 68.230  | 80.575    | 75.852      | 47.831     | 72.074 | 91.658   | 2.013     |
| R34   | Glint360K   | ArcFace | 65.106 | 65.454   | 79.907  | 88.620    | 86.815      | 60.604     | 83.015 | 130.245  | 3.044     |
| R50   | Glint360K   | ArcFace | 70.233 | 69.952   | 85.272  | 91.617    | 90.541      | 66.813     | 87.077 | 166.305  | 4.340     |
| R100  | Glint360K  | ArcFace | 75.567 | 75.202   | 89.488  | 94.285    | 93.434      | 72.528     | 90.659 | 248.590  | 7.038     |
| -       | *Private*     | <div style="width: 50pt">insightface-000 of frvt  | 97.760 | 93.358   | 98.850  | 99.372    | 99.058      | 87.694     | 97.481 | -  | -    |


(MS1M-V2 means MS1M-ArcFace, MS1M-V3 means MS1M-RetinaFace).

Inference time was evaluated on Tesla V100 GPU.



## How to Participate

Send an e-mail to **insightface.challenge(AT)gmail.com** after preparing your onnx model file(without commercial risk), with your name, organization and submission comments.

Some other ways to submit:

1. Submit black-box face feature extracting tool.
    * Use python binding to provide python interface: `feat = get_feature(image, bbox, landmark)`, where shape(image)==(H,W,3), shape(bbox)==(4,), shape(landmark)==(5,2) and shape(feat)==(K,). You can either use the provided landmark or detect them by yourself.
    * In current stage, it should be better to not encrypt your feature embeddings, for fast GPU N:N matrix calculation.
    * You can add some restrictions on your tool. Such as number of api calls and time constraints.


## Leaderboard

[Leaderboard](http://insightface.ai/IFRT) on insightface.ai. (TODO)

TODO
