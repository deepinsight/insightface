## Partial FC
Partial FC is a distributed deep learning training framework for face recognition. The goal of Partial FC is to make 
training large scale classification task (eg. 10 or 100 millions identies) fast and easy. It is faster than model parallel 
and can train more identities.  

![Image text](https://github.com/nttstar/insightface-resources/blob/master/images/partial_speed1.png)

## Celeb-DeepGlint
We clean, merge, and release the **largest** and **cleanest** face recognition dataset **Celeb-DeepGlint**. 
Baseline models trained on Celeb-DeepGlint with our proposed training strategy can easily achieve state-of-the-art. 
The released dataset contains 18 million images of 360K individuals. The performance of **Celeb-DeepGlint** eval on large-scale 
test set IJBC and megaface are as follows:

### IJBC and Megaface
Our backbone is ResNet100, we set feature scale s to 64 and cosine margin m of CosFace at 0.4.
|Test Dataset    | IJBC(TAR@FAR=1e-4) | Megaface(Identification) | Megaface(TAR@FAR=1e-6) |
| :---           | :---:              | :---:                    | :---:                  |
| MS1MV2         | 96.4               | 98.3                     | 98.6                   |
|Celeb-DeepGlint | **97.3**           | **99.1**                 | **99.1**               |

### IFRT
Comming soon.

## Performance
Comming soon.

## Citation
Coming soon.