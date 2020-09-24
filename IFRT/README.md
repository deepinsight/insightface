# InsightFace Recognition Test (IFRT)
**IFRT** is a globalised fair benchmark for face recognition algorithms. IFRT evaluates the algorithm performance on worldwide web pictures which contain various sex, age and race groups, but no identification photos.

**IFRT** testset consists of non-celebrities so we can ensure that it has very few overlap with public available face recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities' photo. As the result, we can evaluate the *FAIR* performance for different algorithms.

Similar to [FRVT](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt), we encourage participants to prepare a black-box feature extractor. To simplify this process, users can just replace their trained ArcFace model(with or without encryption) in our simple open-sourced pre-packaged software.

Submitting features is also allowed, you can send an e-mail to us to request the test image set, with a promise not to redistribute it.

## How to Participate

Send an e-mail to [..] after preparing your black-box feature extractor or image binary features, with your name, organization and submission comments.

There are some ways to submit:

1. (Recommended) Submit black-box face feature extracting tool.
2. (Simplest) Submit your recognition model.
    * Submit MXNet ArcFace model with the same face alignment. In this case, you can just submit the single model file.
    * In other case, such as PyTorch/TF models or ArcFace models with different face alignment method, please give us an example on how to generate feature embeddings. (eg. provide a function `get_feature(image, landmark)`)
3. (Safe) Submit face features. We will give `N` test images named from `0.jpg` to `(N-1).jpg`. Then you should submit a binary feature file with shape `(N,K)` where `K` is your feature dimension. L2 distance will be used on these features.

## Leaderboard

[Leaderboard](http://insightface.ai/IFRT) on insightface.ai. (TODO)

TODO
