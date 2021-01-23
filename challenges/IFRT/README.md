# InsightFace Recognition Test (IFRT)
**IFRT** is a globalised fair benchmark for face recognition algorithms. IFRT evaluates the algorithm performance on worldwide web pictures which contain various sex, age and race groups, but no identification photos.

**IFRT** testset consists of non-celebrities so we can ensure that it has very few overlap with public available face recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. As the result, we can evaluate the FAIR performance for different algorithms.

Similar to [FRVT](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt), we encourage participants to prepare a black-box feature extractor. To simplify this process, users can just replace their trained ArcFace model(with or without encryption) in our simple open-sourced pre-packaged software.

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

TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6).

## Baselines

| Backbone     | Dataset     | DESC.   | African | Caucasian | South Asian | East Asian | ALL   |
| ------------ | ----------- | ------- | ----- | ----- | ------ | ----- | ----- |
| R100         | CASIA       | ArcFace | 39.67 | 53.93 | 47.81  | 16.17 | 37.53 |
| R50          | VGG2        | ArcFace | 49.20 | 65.93 | 56.22  | 27.15 | 47.13 |
| R50          | MS1M-V2     | ArcFace | 71.97 | 83.24 | 79.66  | 22.94 | 56.20 |
| R50          | MS1M-V3     | ArcFace | 76.24 | 86.21 | 84.44  | 37.43 | 71.02 |
| R124         | MS1M-V3     | ArcFace | 81.08 | 89.06 | 87.53  | 38.40 | 74.76 |
| R124         | MS1M-V3     | +FlipTest | 83.22 | 90.43 | 89.22  | 39.61 | 75.69 |
| R100	       | Glint360k   | PartialFC(r=0.1)| 90.45 | 94.60	| 93.96	| 63.91	| 88.23 |
| R180 | *InsightFace-Private*   | ArcFace | 94.45 | 96.98 | 96.02  | 91.67 | 96.26 |
| R180 | *Private*   | block-box | 97.54 | 98.67 | 98.35  | 83.83 | 96.93 |

(MS1M-V2 means MS1M-ArcFace, MS1M-V3 means MS1M-RetinaFace)

(We only consider *African* to *African* comparisons in *African* subset, so others like *African* to *Caucasian* will be ignored)

## How to Participate

Send an e-mail to **insightface.challenge(AT)gmail.com** after preparing your black-box feature extractor or your academic model file(without commercial risk), with your name, organization and submission comments.

There are some ways to submit:

1. (Recommended) Submit black-box face feature extracting tool.
    * Use python binding to provide python interface: `feat = get_feature(image, bbox, landmark)`, where shape(image)==(H,W,3), shape(bbox)==(4,), shape(landmark)==(5,2) and shape(feat)==(K,). You can either use the provided landmark or detect them by yourself.
    * In current stage, it should be better to not encrypt your feature embeddings, for fast GPU N:N matrix calculation.
    * You can add some restrictions on your tool. Such as number of api calls and time constraints.
2. (Simplest) Submit your recognition model.
    * Submit MXNet ArcFace model with the same face alignment. In this case, you can just submit the single model file.
    * In other case, such as PyTorch/TF models or ArcFace models with different face alignment method, please give us an example on how to generate feature embeddings. (eg. provide a function `get_feature(image, bbox, landmark)`)

## Leaderboard

[Leaderboard](http://insightface.ai/IFRT) on insightface.ai. (TODO)

TODO
