# MFR Ongoing

This is the ongoing version of [ICCV-2021 Masked Face Recognition Challenge & Workshop(MFR)](https://ibug.doc.ic.ac.uk/resources/masked-face-recognition-challenge-workshop-iccv-21/). We also extend it to involve some public available and popular benchmarks such as IJBC, LFW, CFPFP and AgeDB.


(:bulb: :bulb: Once you find the name **IFRT** which is *InsightFace Recognition Test* in short anywhere, it is the same as MFR-Ongoing.)


For detail, please check our ICCV 2021 workshop [paper](https://openaccess.thecvf.com/content/ICCV2021W/MFR/papers/Deng_Masked_Face_Recognition_Challenge_The_InsightFace_Track_Report_ICCVW_2021_paper.pdf).

More information about the workshop challenge can be found [here](../iccv21-mfr), for reference.

**MFR** testset consists of non-celebrities so we can ensure that it has very few overlap with public available face recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. As the result, we can evaluate the FAIR performance for different algorithms.

In recent changes, we also add public available popular benchmarks such as IJBC, LFW, CFPFP, AgeDB into **MFR-Ongoing**.


Current submission server link: [http://iccv21-mfr.com/](http://iccv21-mfr.com/)

For any question, please send email to `insightface.challenge AT gmail.com`

## Testsets

In MFR-Ongoing, we will evaluate the accuracy of following testsets:

  * **Accuracy between masked and non-masked faces.**
  * **Accuracy among children(2~16 years old).**
  * **Accuracy of globalised multi-racial benchmarks.**


We ensure that there's no overlap between the above testsets and public available training datasets, as they are not collected from online celebrities.

We also evaluate below public available popular benchmarks:

  * **IJBC under FAR<=e-5 and FAR<=e-4.**
  * **Some 1:1 verification testsets, such as LFW, CFPFP, AgeDB-30.**


### ``Mask test-set:``

Mask testset contains 6,964 identities, 6,964 masked images and 13,928 non-masked images. There are totally 13,928 positive pairs and 96,983,824 negative pairs.

<details>
  <summary>Click to check the sample images(here we manually blur it to protect privacy) </summary>
  <img src="https://github.com/nttstar/insightface-resources/blob/master/images/ifrt_mask_sample.jpg" alt="ifrtsample" width="360">
</details>

### ``Children test-set:``

Children testset contains 14,344 identities and 157,280 images. There are totally 1,773,428 positive pairs and 24,735,067,692 negative pairs.

<details>
  <summary>Click to check the sample images(here we manually blur it to protect privacy) </summary>
  <img src="https://github.com/nttstar/insightface-resources/blob/master/images/ifrt_children_sample.jpg" alt="ifrtsample" width="360">
</details>

### ``Multi-racial test-set (MR in short):``

The globalised multi-racial testset contains 242,143 identities and 1,624,305 images.

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

For multi-racial sets, TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6).

For IJBC and verification test-set, we use the most common test protocal.

Participants are ordered in terms of highest scores across two datasets: **TAR@Mask** and **TAR@MR-All**, by the formula of ``0.25 * TAR@Mask + 0.75 * TAR@MR-All``.




## Baselines

**``2021.04.25``** We made a clean on East Asian subset, by removing children images.

**``2021.04.27``** Add onnx download links.

| Backbone   | Dataset    | Method     | Mask   | Children | African | Caucasian | South Asian | East Asian | All    | size(mb) | infer(ms) | link |
|------------|------------|------------|--------|----------|---------|-----------|-------------|------------|--------|----------|-----------|-----------|
| R100  | Casia  | ArcFace  | 26.623 | 30.359   | 39.666  | 53.933    | 47.807      | 21.572     | 42.735 | 248.904  | 7.073     | [download](https://1drv.ms/u/s!AswpsDO2toNKrUJpk8zC61HVN7Kg?e=zE9JDd) |
| R100  | MS1MV2  | ArcFace  | 65.767 | 60.496   | 79.117  | 87.176    | 85.501      | 55.807     | 80.725 | 248.904  | 7.028     | [download](https://1drv.ms/u/s!AswpsDO2toNKrUTlYEHJCHg3UYM-?e=ihxMpS) |
| R18  | MS1MV3  | ArcFace | 47.853 | 41.047   | 62.613  | 75.125    | 70.213      | 43.859     | 68.326 | 91.658   | 1.856     | [download](https://1drv.ms/u/s!AswpsDO2toNKrTxlT6w1Jo02yzSh?e=KDhFAA) |
| R34  | MS1MV3  | ArcFace | 58.723 | 55.834   | 71.644  | 83.291    | 80.084      | 53.712     | 77.365 | 130.245  | 3.054     | [download](https://1drv.ms/u/s!AswpsDO2toNKrT2O5pgyVtwnjeMq?e=16S8LI) |
| R50  | MS1MV3  | ArcFace | 63.850 | 60.457   | 75.488  | 86.115    | 84.305      | 57.352     | 80.533 | 166.305  | 4.262     | [download](https://1drv.ms/u/s!AswpsDO2toNKrUUWd5i3a5OlFpM_?e=ExBDBN) |
| R100 | MS1MV3 | ArcFace | 69.091 | 66.864   | 81.083  | 89.040    | 88.082      | 62.193     | 84.312 | 248.590  | 7.031     | [download](https://1drv.ms/u/s!AswpsDO2toNKrUPwyqWvNXUlNd3P?e=pTLw9A) |
| R18   | Glint360K   | ArcFace | 53.317 | 48.113   | 68.230  | 80.575    | 75.852      | 47.831     | 72.074 | 91.658   | 2.013     | [download](https://1drv.ms/u/s!AswpsDO2toNKrT5ey4lCqFzlpzDd?e=VWP28J) |
| R34   | Glint360K   | ArcFace | 65.106 | 65.454   | 79.907  | 88.620    | 86.815      | 60.604     | 83.015 | 130.245  | 3.044     | [download](https://1drv.ms/u/s!AswpsDO2toNKrUBcgGkiuUS11Hsd?e=ISGDnP) |
| R50   | Glint360K   | ArcFace | 70.233 | 69.952   | 85.272  | 91.617    | 90.541      | 66.813     | 87.077 | 166.305  | 4.340     | [download](https://1drv.ms/u/s!AswpsDO2toNKrT8jbvHxjqCY0d08?e=igfdrd) |
| R100  | Glint360K  | ArcFace | 75.567 | 75.202   | 89.488  | 94.285    | 93.434      | 72.528     | 90.659 | 248.590  | 7.038     | [download](https://1drv.ms/u/s!AswpsDO2toNKrUFgLEIj-mnkb51b?e=vWqy2q) |
| -       | *Private*     | <div style="width: 50pt">insightface-000 of frvt  | 97.760 | 93.358   | 98.850  | 99.372    | 99.058      | 87.694     | 97.481 | -  | -    |   -  |


(MS1M-V2 means MS1M-ArcFace, MS1M-V3 means MS1M-RetinaFace).

Inference time in above table was evaluated on Tesla V100 GPU, using onnxruntime-gpu==1.6.

## Rules

1. We have two tracks, academic and unconstrained.
2. Please **DO NOT** register the account with messy or random characters(for both username and organization).
3. **For academic submissions, we recommend to set the username as the name of your proposed paper or method. Orgnization hiding is not allowed(or the score will be banned) for this track but you can set the submission as private. You can also create multiple accounts, one account for one method.**
4. Right now we only support 112x112 input, so make sure that the submission model accepts the correct input shape(['*',3,112,112]), in RGB order. Add an interpolate operator into the first layer of the submission model if you need a different input resolution.
5. Participants submit onnx model, then get scores by our online evaluation. 
6. Matching score is measured by cosine similarity.
7. **Online evaluation server uses onnxruntime-gpu==1.8, cuda==11.1, cudnn==8.0.5, GPU is RTX3090.**
8. Any float-16 model weights is prohibited, as it will lead to incorrect model size estimiation.
9. Please use ``onnx_helper.py`` to check whether the model is valid.
10. Leaderboard is now ordered in terms of highest scores across two datasets: **TAR@Mask** and **TAR@MR-All**, by the formula of ``0.25 * TAR@Mask + 0.75 * TAR@MR-All``.



## Submission Guide

1. Participants must package the onnx model for submission using ``zip xxx.zip model.onnx``.
2. Each participant can submit three times a day at most.
3. Please sign-up with the real organization name. You can hide the organization name in our system if you like(not allowed for academic track).
4. You can decide which submission to be displayed on the leaderboard by clicking 'Set Public' button.
5. Please click 'sign-in' on submission server if find you're not logged in.
