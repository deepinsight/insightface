# InsightFace Recognition Test (IFRT)
**IFRT** is a globalised fair benchmark for face recognition algorithms. IFRT evaluates the algorithm performance on worldwide web pictures which contain various sex, age and race groups, but no identification photos.

**IFRT** testset consists of non-celebrities so we can ensure that it has very few overlap with public available face recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. As the result, we can evaluate the FAIR performance for different algorithms.

Similar to [FRVT](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt), we encourage participants to prepare a black-box feature extractor. To simplify this process, users can just replace their trained ArcFace model(with or without encryption) in our simple open-sourced pre-packaged software.

~~Submitting features is also allowed, you can send an e-mail to us to request the test image set, with a promise not to redistribute it.~~

## Dataset Statistics and Visualization

IFRT testset contains 242143 identities and 1624305 images.

## Evaluation Metric

TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6).

## Baselines

| Backbone     | Dataset     | Loss    | BLACK | WHITE | INDIAN | ASIAN | ALL   |
| ------------ | ----------- | ------- | ----- | ----- | ------ | ----- | ----- |
| R100         | CASIA       | ArcFace | 39.67 | 53.93 | 47.81  | 16.17 | 37.53 |
| R50          | MS1M_V2     | ArcFace | 71.97 | 83.24 | 79.66  | 22.94 | 56.20 |
| R50          | MS1M_Retina | ArcFace | 76.24 | 86.21 | 84.44  | 37.43 | 71.02 |
| R124         | MS1M_Retina | ArcFace | -     | -     | -      | -     | -     |
| *NIST-Level* | *Private*   | ArcFace | 92.26 | 95.73 | 93.97  | 91.67 | 94.76 |

## How to Participate

Send an e-mail to guojia(AT)gmail.com after preparing your black-box feature extractor or your trained academic model file(without commercial risk), with your name, organization and submission comments.

There are some ways to submit:

1. (Recommended) Submit black-box face feature extracting tool.
2. (Simplest) Submit your recognition model.
    * Submit MXNet ArcFace model with the same face alignment. In this case, you can just submit the single model file.
    * In other case, such as PyTorch/TF models or ArcFace models with different face alignment method, please give us an example on how to generate feature embeddings. (eg. provide a function `get_feature(image, bbox, landmark)`)

## Leaderboard

[Leaderboard](http://insightface.ai/IFRT) on insightface.ai. (TODO)

TODO
