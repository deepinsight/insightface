# [OSTeC: One-Shot Texture Complition](https://openaccess.thecvf.com/content/CVPR2021/html/Gecer_OSTeC_One-Shot_Texture_Completion_CVPR_2021_paper.html)
#### [CVPR 2021]
[![arXiv Prepring](https://img.shields.io/badge/arXiv-Preprint-lightgrey?logo=arxiv)](https://arxiv.org/abs/2012.15370)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


 [Baris Gecer](http://barisgecer.github.io) <sup> 1,2</sup>, [Jiankang Deng](https://jiankangdeng.github.io/) <sup> 1,2</sup>, & [Stefanos Zafeiriou](https://wp.doc.ic.ac.uk/szafeiri/) <sup> 1,2</sup>
 <br/>
 <sup>1 </sup>Imperial College London
 <br/>
 <sup>2 </sup> Huawei CBG
 <br/>

<p align="center">
<img width="12%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/gifs/im18.gif" style="background-color:white; display: inline;" />
<img width="12%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/gifs/norah.gif" style="background-color:white; display: inline;" />
<img width="12%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/gifs/robin.gif" style="background-color:white; display: inline;" />
<img width="12%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/gifs/teaser2.gif" style="background-color:white; display: inline;" />
<img width="12%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/gifs/teaser3.gif" style="background-color:white; display: inline;" />
<img width="12%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/gifs/teaser4.gif" style="background-color:white; display: inline;" />
<img width="12%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/gifs/teaset5.gif" style="background-color:white; display: inline;" />
</p>
<p align="center"><img width="100%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/teaser.jpg" style="background-color:white;" /></p>

## Abstract

The last few years have witnessed the great success of non-linear generative models in synthesizing high-quality photorealistic face images. Many recent 3D facial texture reconstruction and pose manipulation from a single image approaches still rely on large and clean face datasets to train image-to-image Generative Adversarial Networks (GANs). Yet the collection of such a large scale high-resolution 3D texture dataset is still very costly and difficult to maintain age/ethnicity balance. Moreover, regression-based approaches suffer from generalization to the in-the-wild conditions and are unable to fine-tune to a target-image. In this work, we propose an unsupervised approach for one-shot 3D facial texture completion that does not require large-scale texture datasets, but rather harnesses the knowledge stored in 2D face generators. The proposed approach rotates an input image in 3D and fill-in the unseen regions by reconstructing the rotated image in a 2D face generator, based on the visible parts. Finally, we stitch the most visible textures at different angles in the UV image-plane. Further, we frontalize the target image by projecting the completed texture into the generator. The qualitative and quantitative experiments demonstrate that the completed UV textures and frontalized images are of high quality, resembles the original identity, can be used to train a texture GAN model for 3DMM fitting and improve pose-invariant face recognition.

## Overview

<p align="center"><img width="100%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/overview.jpg" style="background-color:white;" /></p>
Overview of the method. The proposed approach iteratively optimizes the texture UV-maps for different re-rendered images with their masks. At the end of each optimization, generated images are used to acquire partial UV images by dense landmarks. Finally, the completed UV images are fed to the next iteration for progressive texture building.
<br/>


## Requirements
**This implementation is only tested under Ubuntu environment with Nvidia GPUs and CUDA 10.0 and CuDNN-7.0 installed.**

## Installation
### 1. Clone the repository and set up a conda environment as follows:
```
git clone https://github.com/barisgecer/OSTeC --recursive
cd OSTeC
conda env create -f environment.yml -n ostec
source activate ostec
```

### 2. Installation of Deep3DFaceRecon_pytorch
- **2.a.** Install Nvdiffrast library:
```
cd external/deep3dfacerecon/nvdiffrast    # ./OSTeC/external/deep3dfacerecon/nvdiffrast 
pip install .
```
- **2.b.** Install Arcface Pytorch:
```
cd ..    # ./OSTeC/external/deep3dfacerecon/
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch/ ./models/
```

- **2.c.** Prepare prerequisite models: Deep3DFaceRecon_pytorch method uses [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) to represent 3d faces. Get access to BFM09 using this [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access, download "01_MorphableModel.mat". In addition, we use an Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace). Download the Expression Basis (Exp_Pca.bin) using this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing). Organize all files into the following structure:
```
OSTeC
│
└─── external
     │
     └─── deep3dfacerecon
          │
          └─── BFM
              │
              └─── 01_MorphableModel.mat
              │
              └─── Exp_Pca.bin
              |
              └─── ...
```
- **2.d.** Deep3DFaceRecon_pytorch provides a model trained on a combination of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 
[LFW](http://vis-www.cs.umass.edu/lfw/), [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm),
[IJB-A](https://www.nist.gov/programs-projects/face-challenges), [LS3D-W](https://www.adrianbulat.com/face-alignment), and [FFHQ](https://github.com/NVlabs/ffhq-dataset) datasets. Download the pre-trained model using this [link (google drive)](https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP?usp=sharing) and organize the directory into the following structure:
```
OSTeC
│
└─── external
     │
     └─── deep3dfacerecon
          │
          └─── checkpoints
               │
               └─── face_recon
                   │
                   └─── epoch_latest.pth

```
### 3. Download Face Recognition \& Landmark Detection \& VGG \& Style-Encoder models
- Download the models here: https://drive.google.com/file/d/1TBoNt55vleRkMZaT9XKt6oNQmo8hkN-Q/view?usp=sharing

- And place it under 'models' directory like the following:
```
OSTeC
│
└─── models
     │
     └─── resnet_18_20191231.h5
     │
     └─── vgg16_zhang_perceptual.pkl
     │
     └─── alignment
     │         .
     │         .
     │
     └─── fr_models
               .
               .

```

### 4. Download Topology info files
- Download the topology files here: https://drive.google.com/file/d/1mvb2uDMPNGL1MlBgP6Op00gPdEMQUUWb/view?usp=sharing

- And place it under 'models/topology' directory like the following:
```
OSTeC
│
└─── models
     │         .
     │         .
     │         .
     │
     └─── topology
          │
          └─── trilist.pkl
          │
          └─── tcoords.pkl
                    .
                    .

```



### 5. Download Face Segmentation models
- Download the Graphonomy model here: https://drive.google.com/file/d/1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9/view?usp=sharing
(If the link doesn't work for some reason check the original [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) github page and download 'CIHP trained model')

- And place it under 'models' directory like the following:
```
OSTeC
│
└─── models
     │
     └─── Graphonomy
         │
         └─── inference.pth
```

<!--- ### 4. Download StyleGANv2 model
- Download the model from the original repo: https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl
And place it under 'models' directory like the following:
```
OSTeC
│
└─── models
     │
     └─── stylegan2_networks_stylegan2-ffhq-config-f

```
-->

## Usage
- Run ```python run_ostec.py --source_dir [source_dir] --save_dir [save_dir] [-f] -i [iterations (default 200)] -m [soft|hard|auto]```
- Modes (-m or --mode):
   * soft: keep the original texture for visible parts (recommended when the input image is high resolution, near-frontal, and non-occluded.)
   * hard: generate all
   * auto: soft for frontal, hard for profile images

## More Results

<p align="center"><img width="100%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/comp2.jpg" /></p>
<p align="center"><img width="100%" src="https://raw.githubusercontent.com/barisgecer/OSTeC/main/figures/comp1.jpg" /></p>
<br/>

## License
- The source code shared here is protected under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License which does **NOT** allow commercial  use. To view a copy of this license, see LICENSE
- Copyright (c) 2020, Baris Gecer. All rights reserved.
- This work is made available under the CC BY-NC-SA 4.0.


## Acknowledgement
- Our projection relies on NVIDIA's [StyleGANv2](https://github.com/NVlabs/stylegan2)
- Thanks [@jiankangdeng](https://jiankangdeng.github.io/) for providing Face Recognition and Landmark Detection models
- We use [MTCNN](https://github.com/ipazc/mtcnn) for face detection
- We use [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) for face segmentation (i.e. to exclude hairs, occlusion)
- 3D face reconstruction has been originally solved by [GANFit](https://github.com/barisgecer/GANFit). However, since it is commercialized and will not be public, I had to re-implement the ports for [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch).
- We initialize StyleGAN parameters by [Style-Encoder](https://github.com/rolux/stylegan2encoder/issues/2) (by [@rolux](https://github.com/rolux), [@pbaylies](https://github.com/pbaylies)).
- Thanks [Zhang et al.](https://richzhang.github.io/PerceptualSimilarity/) for VGG16 model

## Citation
If you find this work is useful for your research, please cite our paper: 

```
@InProceedings{Gecer_2021_CVPR,
    author    = {Gecer, Baris and Deng, Jiankang and Zafeiriou, Stefanos},
    title     = {OSTeC: One-Shot Texture Completion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {7628-7638}
}
```
<br/>
