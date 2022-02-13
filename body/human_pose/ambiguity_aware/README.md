# Towards Alleviating the Modeling Ambiguity of Unsupervised Monocular 3D Human Pose Estimation

## Introduction

**Ambiguity-Aware** studies the ambiguity problem in the task of unsupervised 3D human pose estimation from 2D counterpart, please refer to [ICCV2022](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Towards_Alleviating_the_Modeling_Ambiguity_of_Unsupervised_Monocular_3D_Human_ICCV_2021_paper.pdf) for more details.


<div align="center">
 <img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/all.gif" alt="videovis" width="800">
</div>


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

This part will be updated soon.
## Inference 
We put some samples with preprocessed 2d keypoints at `scripts/demo_input`. Run inference with command `sh demo.sh` and output can be found at `scripts/demo_output`. 

## Evaluation
### Evaluation on Human3.6M 
##### 2D ground-truth as inputs
* baseline `python main.py --cfg ../cfg/h36m_gt_adv.yaml --pretrain ../models/adv.pth.tar --gpu 0 --eval `
* scale `python main.py --cfg ../cfg/h36m_gt_scale.yaml --pretrain ../models/tmc_klbone.pth.tar  --eval --gpu 0`

##### 2D predictions as inputs
* baseline `python main.py --cfg ../cfg/pre_adv.yaml --pretrain ../models/pre_adv.pth.tar --gpu 0 --eval `
* scale `python main.py --cfg ../cfg/pre_tmc_klbone.yaml --pretrain ../models/pre_tmc_klbone.pth.tar --gpu 0 --eval `

**Note:** baseline is our reproduced version fo "Unsupervised 3d pose estimation with geometric self-supervision"

### Evaluation on LSP
use the pretrained model from Human3.6M

`python eval_lsp.py --cfg ../cfg/h36m_gt_scale.yaml --pretrain ../models/tmc_klbone.pth.tar`

### Results

The expected **MPJPE** and **P-MPJPE**  results on **Human36M** dataset are shown here:

| Input  | Model                         |     MPJPE     |     PMPJPE     | 
| :--------- | :------------                  | :------------: | :------------: | 
| GT | baseline                              |      105.0      |       46.0    |   
| GT | best                             |      87.85      |       42.0     |     
| Pre | baseline                             |      113.3     |    54.9     | 
| Pre | best                            |      93.1       |    52.3     | 


**Note:**  MPJPE from the evaluation is slightly different from the performance we release in the paper. This is because MPJPE in the paper is the best MPJPE during training process.



## Training 
### Human3.6M 
* Using ground-truth 2D as inputs: 
    
    baseline `python main.py --cfg ../cfg/h36m_gt_adv.yaml --gpu 0 `

    best `python main.py --cfg ../cfg/h36m_gt_scale.yaml --gpu 0`

* Using predicted 2D as inputs: 

    baseline `python main.py --cfg ../cfg/pre_adv.yaml --gpu 0 `

    best `python main.py --cfg ../cfg/pre_tmc_klbone.yaml --gpu 0`

## Visualization

### Human3.6M
<div align="center">
 <img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/S9_Discussion 1.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/S9_Phoning 1.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/S9_Photo.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/S9_WalkTogether 1.gif" width="200"/>
</div>

### Sureal
<div align="center">
 <img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/surreal1.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/surreal2.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/surreal3.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/surreal4.gif" width="200"/>
</div>

### MPI-3DHP
<div align="center">
 <img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/TS1.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/TS2.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/TS3.gif" width="200"/><img src="https://github.com/yuzhenbo/yuzhenbo.github.io/raw/main/assets/extra/ICCV2022/TS6.gif" width="200"/>
</div>


### The code of our another paper in ICCV2022 Skeleton2Mesh will be coming soon! 
