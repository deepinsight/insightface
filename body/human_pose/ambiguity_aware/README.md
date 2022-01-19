# Towards Alleviating the Modeling Ambiguity of Unsupervised Monocular 3D Human Pose Estimation

## Introduction

**Ambiguity-Aware-HPE** studies the ambiguity problem in the task of unsupervised 3D human pose estimation from 2D counterpart, which is initially proposed in [CVPR-2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Towards_Alleviating_the_Modeling_Ambiguity_of_Unsupervised_Monocular_3D_Human_ICCV_2021_paper.pdf). 

<img src=https://github.com/yuzhenbo/yuzhenbo.github.io/tree/main/assets/extra/ICCV2022/all.gif width="760" />


## Installation 
```
conda create -n uvhpe python=3.6 
conda activate uvhpe
pip install -r requirements.txt
# for output,  tensorboard, visualization  
mkdir log output vis models data
```

## Dataset And Pretrained Models 
Download our preprocessed dataset into `data` and pretrained models into `models` from [webpage](https://sites.google.com/view/ambiguity-aware-hpe)

## Evaluation
### Human3.6M 
##### 2D ground-truth as inputs
* adv `python main.py --cfg ../cfg/h36m_gt_adv.yaml --pretrain ../models/adv.pth.tar --gpu 0 --eval `
* scale `python main.py --cfg ../cfg/h36m_gt_scale.yaml --pretrain ../models/tmc_klbone.pth.tar  --eval --gpu 0`

##### 2D predictions as inputs
* adv `python main.py --cfg ../cfg/pre_adv.yaml --pretrain ../models/pre_adv.pth.tar --gpu 0 --eval `
* scale `python main.py --cfg ../cfg/pre_tmc_klbone.yaml --pretrain ../models/pre_tmc_klbone.pth.tar --gpu 0 --eval `

## Inference On LSP
use the pretrained model from Human3.6M

`python eval_lsp.py --cfg ../cfg/h36m_gt_scale.yaml --pretrain ../models/tmc_klbone.pth.tar`

### Results

The expected **MPJPE** and **P-MPJPE**  results on **Human36M** dataset are shown here:

| Input  | Model                         |     MPJPE     |     PMPJPE     | 
| :--------- | :------------                  | :------------: | :------------: | 
| GT | adv                              |      105.0      |       46.0    |   
| GT | best                             |      87.85      |       42.0     |     
| Pre | adv                             |      113.3     |    54.9     | 
| Pre | best                            |      93.1       |    52.3     | 


**Note:  MPJPE from the evaluation is slightly different from the performance we release in the paper. This is because MPJPE in the paper is the best MPJPE during training process** 

## Inference 
We put some samples with preprocessed 2d keypoints at `scripts/demo_input`. Run inference with command `sh demo.sh` and output can be found at `scripts/demo_output`. 


## Training 
### Human3.6M 
* Using ground-truth 2D as inputs: 
    
    adv `python main.py --cfg ../cfg/h36m_gt_adv.yaml --gpu 0 `

    best `python main.py --cfg ../cfg/h36m_gt_scale.yaml --gpu 0`

* Using predicted 2D as inputs: 

    adv `python main.py --cfg ../cfg/pre_adv.yaml --gpu 0 `

    best `python main.py --cfg ../cfg/pre_tmc_klbone.yaml --gpu 0`
    
