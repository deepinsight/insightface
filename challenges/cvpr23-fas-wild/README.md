# 4th Face Anti-spoofing Workshop and Challenge@CVPR2023, Wild Track

<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/faswild_large.png" width="640"/>
</div>

## Updates


## Challenge

We host the WILD track of Face Anti-spoofing Workshop and Challenge@CVPR2023 here. The challenge will officially start together with [4th Face Anti-spoofing Workshop](https://sites.google.com/view/face-anti-spoofing-challenge/welcome/challengecvpr2023). [Registration](#) will open soon on codalab.

Our competition encompasses over 800K spoof photos and over 500K live photos. In the spoof photos, there are three major categories and 17 subcategories.

### Rules and Regulations

1) Any extra data or pretrained model trained from extra data cannot be used in this challenge.

2) Only one DL model can be used, we can not accept the fusion results from many DL models. The computational cost of a single DL model should be **less than 5G FLOPs**. 

3) The top-3 winners are required to submit the code for the entire method, ensuring reproducibility of the results and compliance with all contest rules, otherwise the score will be disqualified.

## Evaluation

### Evaluation Criteria

For the performance evaluation, we selected the recently standardized ISO/IEC 30107-3 metrics: Attack Presentation Classification Error Rate (APCER), Normal/Bona Fide Presentation Classification Error Rate (NPCER/BPCER) and Average Classification Error Rate (ACER) as the evaluation metric, in which APCER and BPCER/NPCER are used to measure the error rate of fake or live samples, respectively. The ACER value is used as the final evaluation criterion.


### Submission Format

In order to submit results at one time, participants need to combine the dev and test predictions into one file before result submission via codalab system. Note that the order of the samples cannot be changed and the dev sample list needs to be written before the test samples.

The final submission file contains a total of 895,237 lines. Each line in the file contains two parts separated by a space. Such as: 
```
dev/000001.jpg 0.15361                   #Note:  line 1- the first row of dev.txt

......

dev/140058.jpg 0.23394                   #Note:  line 140,058 the last row of dev.txt
test/000001.jpg 0.15361                   #Note:  line 140,059 the first row of test.txt  

......   

test/755179.jpg 0.23394                   #Note:  line 895,237 the last row of test.txt
```

## Dataset

### Rules

1. The dataset and its subsets can only be used for academic research purposes.
2. The user is not allowed to use the dataset or its subsets for any type of commercial purpose.
3. Any form of usage of the dataset in defamatory, pornographic, or any other unlawful manner, or violation of any applicable regulations or laws is forbidden. We are not responsible for any consequences caused by the above behaviors.
4. The User is not allowed to distribute, broadcast, or reproduce the dataset or its subsets in any way without official permission.
5. The user is not allowed to share, transfer, sell or resell the dataset or its subsets to any third party for any purpose. HOWEVER, providing the dataset access to user’s research associates, colleagues or team member is allowed if user’s research associates, colleagues or team member agree to be bound by these usage rules.
6. All images in this dataset can be used for academic research purposes, BUT, only the approved images of the dataset can be exhibited on user’s publications(including but not limited to research paper, presentations for conferences or educational purposes). The approved images have special marks and are listed in a appendix.
7. We reserve the right to interpret and amend these rules.
8. please cite us if the InsightFace Wild Anti-Spoofing Dataset or its subset is used in your research:
```
@misc{2023insightfacefaswild,
    title={InsightFace Wild Anti-Spoofing Dataset},
    author={Dong Wang, Qiqi Shao, Haochi He, Zhian Chen, Jia Guo, Jiankang Deng, Jun Wan, Ajian Liu, Sergio Escalera, Hugo Jair Escalante, Isabelle Guyon, Zhen Lei, Chenxu Zhao, Shaopeng Tang },
    howpublished = {\url{https://github.com/deepinsight/insightface/tree/master/challenges/cvpr23-fas-wild}},
    year={2023}
}
```



### Download

All users can obtain and use this dataset and its subsets only after signing the [Agreement](https://github.com/nttstar/insightface-resources/raw/master/files/License%20Agreement%20for%20InsightFace%20Wild%20Anti-Spoofing%20Dataset.pdf) and sending it to the official e-mail ``insightface.challenge#gmail.com``.


### Dataset Annotations

Please refer to the following table for detailed information on the number of labeled data and examples in the dataset:

#### Spoofing Images

1) Training Subset, live/spoof labels and categorization information are given:

<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/faswild_train_dataset.png" width="1024"/>
</div>



2) Dev and Test Subsets, where dev set is used to select the threshold.

<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/faswild_devtest_dataset.png" width="1024"/>
</div>

#### Live Images:

There're 205,146 live images in training dataset, and 51,299/273,126 images in dev and test datasets respectively.


## Baselines


## Feedback

1) If you have any questions regarding the challenge, kindly open an issue on insightface github. (recommended)
2) Or you can send an e-mail to ``insightface.challenge#gmail.com``

