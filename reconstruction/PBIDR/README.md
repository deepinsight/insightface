# Facial Geometric Detail Recovery via Implicit Representation

:herb: **[Facial Geometric Detail Recovery via Implicit Representation](https://arxiv.org/abs/2203.09692)**

Xingyu Ren, Alexandros Lattas, Baris Gecer, Jiankang Deng, Chao Ma, Xiaokang Yang, and Stefanos Zafeiriou. 

*arXiv Preprint 2022*

## Introduction

![overview](https://github.com/deepinsight/insightface/blob/master/reconstruction/PBIDR/figures/overview.jpg)

This paper introduces a single facial image geometric detail recovery algorithm. The method generates complete high-fidelity texture maps from occluded facial images, and employs implicit renderer and shape functions, to derive fine geometric details by decoupled specular normals. As a bonus, it disentangles the facial texture into approximate diffuse albedo, diffuse and specular shading in a self-supervision manner.

## Installation

Please refer to the installation and usage of [IDR](https://github.com/lioryariv/idr).

The code is compatible with python 3.7 and pytorch 1.7.1. In addition, the following packages are required:  
numpy, scikit-image, trimesh (with pyembree), opencv, torchvision, pytorch3d 0.4.0.

You can see [INSTALL.md](INSTALL.md) for manual installation.

## Tutorial

### Data Preprocessing

 We have provided several textured meshes from [Google Drive](https://drive.google.com/file/d/1R7MdWawdMSjQUOnciJ5mb1pcwoY61Tzc/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/16mAqB_7mlbW2--0__patWA) (password: wp47). Otherwise, please refer to [OSTeC](https://github.com/barisgecer/OSTeC) to make a textured mesh firstly.

Please download raw textured meshes and run:

 ```shell
cd ./code
bash script/data_process.sh
 ```

 You can synthesize the auxiliary image sets for the next implicit details recovery.

### Train & Eval

You can start the training phase with the following script.

 ```shell
cd ./code
bash script/fast_train.sh
 ```

 We also provide a script for eval: 

 ```shell
cd ./code
bash script/fast_eval.sh
 ```

## Citation

 If any parts of our paper and codes are helpful to your work, please generously citing:

 ```
@misc{ren2022facial,
      title={Facial Geometric Detail Recovery via Implicit Representation}, 
      author={Xingyu Ren and Alexandros Lattas and Baris Gecer and Jiankang Deng and Chao Ma and Xiaokang Yang and Stefanos Zafeiriou},
      year={2022},
      eprint={2203.09692},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 ```

## Reference

 We refer to the following repositories when implementing our whole pipeline. Thanks for their great work.

 - [barisgecer/OSTeC](https://github.com/barisgecer/OSTeC)
 - [lioryariv/idr](https://github.com/lioryariv/idr)
