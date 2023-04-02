# Physical-based Rendering for NIR-VIS Face Recognition

by [Yunqi Miao*](https://yoqim.github.io/), [Alexandros Lattas*](https://alexlattas.com/), [Jiankang Deng](https://jiankangdeng.github.io/), [Jungong Han](https://jungonghan.github.io/), and [Stefanos Zafeiriou](https://wp.doc.ic.ac.uk/szafeiri/).


For more information, please check our

**[[Arxiv]](https://arxiv.org/abs/2211.06408)**
**[[NeurIPS 2022 Paper]](https://openreview.net/pdf?id=NN_TpS5dpo5)**


If you find this project useful in your research, please consider citing:

```
@inproceedings{miao2022physically,
  title={Physically-Based Face Rendering for NIR-VIS Face Recognition},
  author={Miao, Yunqi and Lattas, Alexandros and Deng, Jiankang and Han, Jungong and Zafeiriou, Stefanos},
  booktitle={NeurIPS 2022},
  year={2022}
}
```

# Overview
![poster](https://github.com/yoqim/PR-HFR/blob/main/pics/Poster.png)

# Training

For this project, we used python 3.7.10.

## How to run?

```shell
sh run.sh
```


# Testing
## Preparation
- Downloading data (112 x 112) from [[Google drive]](https://drive.google.com/file/d/1Smd-Bdwj4tCbNugmoa66vxnJAU613bCo/view?usp=sharing)
   - Put data to `data/$dataset_name` 
   - Copy data list from [here](https://github.com/yoqim/PR-HFR/tree/main/data)

>Note that: casia(fold_1) is provided for research purposes only. For the rest data, please refer to the original publications.


- Downloading models from [[Google drive]](https://drive.google.com/file/d/1XjlnvbXmRD5xLJo7lLTy8LyQbMYRoz8C/view?usp=sharing)
    - Put pretrain model at `models/pretrain/`
    - Put finetune model at `models/finetune/$dataset/`


## How to run?

```shell
sh eval.sh
```
