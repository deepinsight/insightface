
# *InsightFace* : Implementation for paper 'ArcFace: Additive Angular Margin Loss for Deep Face Recognition'

  Paper by Jiankang Deng, Jia Guo, and Stefanos Zafeiriou

### License

   InsightFace is released under the MIT License.

### Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Installation](#installation)
0. [Usage](#usage)
0. [Models](#models)
0. [Results](#results)
0. [Contribution](#contribution)
0. [Contact](#contact)


### Introduction

   Paper link: [here](https://arxiv.org/abs/1801.07698). 
   
   This repository contains the entire pipeline for deep face recognition with **`ArcFace`** and other popular methods including Softmax, Triplet Loss, SphereFace and AMSoftmax/CosineFace, etc..

   **ArcFace** is a recently proposed face recognition method. It was initially described in an [arXiv technical report](https://arxiv.org/abs/1801.07698). By using ArcFace and this repository, you can simply achieve LFW 99.80+ and Megaface 98%+ by a single model.

   We provide a refined MS1M dataset for training here, which was already packed in MXNet binary format. It allows researcher or industrial engineer to develop a deep face recognizer quickly by only two stages: 1. Download binary dataset; 2. Run training script.

   In InsightFace, we support several popular network backbones and can be set just in one parameter. Below is the list until today:

- ResNet

- MobiletNet

- InceptionResNetV2

- DPN

- DenseNet

 In our paper, we found there're overlap identities between facescrub dataset and Megaface distractors which greatly affects the identification performance. Sometimes more than 10 percent improvement can be achieved after removing these overlaps. This list will be made public soon in this repository.


   ArcFace achieves the state-of-the-art identification performance in MegaFace Challenge, at 98%+. 


### Citation

   If you find **InsightFace/ArcFace** useful in your research, please consider to cite our paper.
   
```
@misc{arcface2018,
  author =       {Jiankang Deng, Jia Guo and Stefanos Zafeiriou},
  title =        {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  journal =      {arXiv preprint arXiv:1801.07698},
  year =         {2018}
}
```

### Requirements
      1. The only requirement is `MXNet` with GPU support(Python 2.7).

### Installation
   1. Install MXNet by 

       ```
       pip install mxnet-cu80
       ```

     

   2. Clone the InsightFace repository. We'll call the directory that you cloned InsightFace as **`INSIGHTFACE_ROOT`**.

       ```Shell
       git clone --recursive https://github.com/deepinsight/insightface.git
       ```
      


### Usage

   *After successfully completing the [installation](#installation)*, you are ready to run all the following experiments.

   #### Part 1: Dataset Downloading.
   **Note:** In this part, we assume you are in the directory **`$INSIGHTFACE_ROOT/`**
   1. Download the training set (`MS1M`) from [here] and place them in **`datasets/`**. Each training dataset includes following 7 files:

      ```Shell
      	- train.idx
      	- train.rec
      	- property
      	- lfw.bin
      	- cfp_ff.bin
      	- cfp_fp.bin
      	- agedb_30.bin
      ```
       The first three files are the dataset itself while the last four ones are binary verification sets.

   #### Part 2: Train
   **Note:** In this part, we assume you are in the directory **`$INSIGHTFACE_ROOT/src/`**. Before start  any training procedure, make sure you set the correct env params for MXNet to ensure the performance.

```
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
```

 Now we give some examples below. Our experiments were all done on Tesla P40 GPU.

   1. Train ArcFace with LResNet100E-IR.

      ```Shell
      CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 --data-dir ../datasets/faces_ms1mr_112x112  --prefix ../model-r100-arcface
      ```
      It will output verification results of *LFW*, *CFP-FF*, *CFP-FP* and *AgeDB-30* every 2000 batches. You can check all command line options in **train\_softmax.py**.

      This model can achieve **LFW 99.80+ and MegaFace 98.0%+**

   2. Train AMSoftmax/CosineFace with LResNet50E-IR.

      ```Shell
      CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r50 --loss-type 2 --margin-m 0.35 --data-dir ../datasets/faces_ms1mr_112x112 --prefix ../model-r50-amsoftmax
      ```

   3. Train Softmax with LMobileNetE.

      ```Shell
      CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network m1 --loss-type 0 --data-dir ../datasets/faces_ms1mr_112x112 --prefix ../model-m1-softmax
      ```

4. Re-Train with Triplet on above Softmax model.
   ```Shell
   CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network m1 --loss-type 12 --lr 0.005 --mom 0.0 --per-batch-size 150 --data-dir ../datasets/faces_ms1mr_112x112 --pretrained ../model-m1-softmax,50 --prefix ../model-m1-triplet
   ```

5. Train Softmax with LDPN107E.

      ```Shell
      CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_softmax.py --network p107 --loss-type 0 --per-batch-size 64 --data-dir ../datasets/faces_vgg_112x112 --prefix ../model-p107-softmax
      ```



   #### Part 3: MegaFace Test

   **Note:** In this part, we assume you are in the directory **`$INSIGHTFACE_ROOT/src/megaface/`**

   1. Align all face images of facescrub dataset and megaface distractors. Please check the alignment scripts under **`$INSIGHTFACE_ROOT/src/align/`**. (We may plan to release these data soon, not sure.)

   2. Next, generate feature files for both facescrub and megaface images.

      ```Shell
      python -u gen_megaface.py
      ```

   3. Remove Megaface noises which generates new feature files.

      ```Matlab
      python -u remove_noises.py
      ```
   4. Start to run megaface development kit to produce final result. 

### Models
      1. We plan to make some models public soon.

### Results
   
   We simply report the performance of **LResNet100E-IR** network trained on **MS1M** dataset with **ArcFace** loss.

| Method  | LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | MegaFace1M(%) |
| ------- | ------ | --------- | --------- | ----------- | ------------- |
| ArcFace | 99.80+ | 99.85+    | 94.0+     | 97.90+      | **98.0+**     |



### Contribution
   - Any type of PR or third-party contribution are welcome.

### Contact

     [Jia Guo](guojia[at]gmail.com) and [Jiankang Deng](https://ibug.doc.ic.ac.uk/people/jdeng)

     Questions can also be left as issues in the repository. We will be happy to answer them.
   ```

   ```
