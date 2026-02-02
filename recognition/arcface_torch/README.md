# Distributed ArcFace Training in PyTorch

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-agedb-30)](https://paperswithcode.com/sota/face-verification-on-agedb-30?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-cfp-fp)](https://paperswithcode.com/sota/face-verification-on-cfp-fp?p=killing-two-birds-with-one-stone-efficient)

This is the official PyTorch implementation of ArcFace (Additive Angular Margin Loss for Deep Face Recognition). The repository provides efficient distributed training implementation with support for large-scale datasets.

## Overview

ArcFace-Torch implements state-of-the-art face recognition training with the following capabilities:

- Distributed training across multiple GPUs and nodes
- Mixed precision training with automatic mixed precision (AMP)
- Partial FC for efficient training on datasets with millions of identities
- Support for both CNN backbones (ResNet, MobileFaceNet) and Vision Transformers
- Training on large-scale datasets (WebFace42M with 42.5M images, Glint360K with 17.1M images)
- Built-in ONNX export for deployment

## Requirements

- Python >= 3.7
- PyTorch >= 1.12.0
- NVIDIA DALI (optional, for faster data loading)

## Installation

```bash
git clone https://github.com/deepinsight/insightface.git
cd insightface/recognition/arcface_torch
pip install -r requirement.txt
```

For DALI installation, refer to [docs/install_dali.md](docs/install_dali.md).

## Training

### Single GPU

```bash
python train_v2.py configs/ms1mv3_r50_onegpu
```

Note: Single GPU training is primarily for testing. Multi-GPU training is recommended for production use.

### Multi-GPU (Single Node)

```bash
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50
```

### Multi-Node Training

On each node:

```bash
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12581 train_v2.py configs/wf42m_pfc02_16gpus_r100

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=12581 train_v2.py configs/wf42m_pfc02_16gpus_r100
```

### Vision Transformer

```bash
torchrun --nproc_per_node=8 train_v2.py configs/wf42m_pfc03_40epoch_8gpu_vit_b
```

## Datasets

### Available Datasets

| Dataset | Identities | Images | Link |
|---------|-----------|---------|------|
| MS1MV2 | 87K | 5.8M | [Download](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) |
| MS1MV3 | 93K | 5.2M | [Download](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) |
| Glint360K | 360K | 17.1M | [Download](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) |
| WebFace42M | 2M | 42.5M | [Preparation Guide](docs/prepare_webface42m.md) |

### Custom Dataset

See [docs/prepare_custom_dataset.md](docs/prepare_custom_dataset.md) for instructions on preparing your own dataset.

### Data Preprocessing

For DALI users, shuffle the rec files before training:

```bash
python scripts/shuffle_rec.py ms1m-retinaface-t1
```

## Model Zoo

Pre-trained models are available for non-commercial research purposes only.

- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g) (Password: e8pw)
- [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

## Performance

### Evaluation

Performance is evaluated on IJB-C and ICCV2021-MFR benchmarks. MFR-ALL contains 242,143 identities and 1,624,305 images with TAR measured at FAR < 1e-6.

### Single-Host GPU Training Results

| Dataset | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Training Log |
|---------|----------|---------|-------------|-------------|--------------|
| MS1MV2 | mobilefacenet | 62.07 | 93.61 | 90.28 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_mbf/training.log) |
| MS1MV2 | r50 | 75.13 | 95.97 | 94.07 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r50/training.log) |
| MS1MV2 | r100 | 78.12 | 96.37 | 94.27 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r100/training.log) |
| MS1MV3 | mobilefacenet | 63.78 | 94.23 | 91.33 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_mbf/training.log) |
| MS1MV3 | r50 | 79.14 | 96.37 | 94.47 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r50/training.log) |
| MS1MV3 | r100 | 81.97 | 96.85 | 95.02 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r100/training.log) |
| Glint360K | mobilefacenet | 70.18 | 95.04 | 92.62 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_mbf/training.log) |
| Glint360K | r50 | 86.34 | 97.16 | 95.81 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r50/training.log) |
| Glint360K | r100 | 89.52 | 97.55 | 96.38 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r100/training.log) |
| WF4M | r100 | 89.87 | 97.19 | 95.48 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf4m_r100/training.log) |
| WF12M-PFC-0.2 | r100 | 94.75 | 97.60 | 95.90 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc02_r100/training.log) |
| WF12M-PFC-0.3 | r100 | 94.71 | 97.64 | 96.01 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc03_r100/training.log) |
| WF12M | r100 | 94.69 | 97.59 | 95.97 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_r100/training.log) |
| WF42M-PFC-0.2 | r100 | 96.27 | 97.70 | 96.31 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_r100/training.log) |
| WF42M-PFC-0.2 | ViT-T | 92.04 | 97.27 | 95.68 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_40epoch_8gpu_vit_t/training.log) |
| WF42M-PFC-0.3 | ViT-B | 97.16 | 97.91 | 97.05 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_8gpu/training.log) |

### Multi-Host GPU Training Results

| Dataset | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Throughput | Training Log |
|---------|----------|---------|-------------|-------------|-----------|--------------|
| WF42M-PFC-0.2 | r50(512x8) | 93.83 | 97.53 | 96.16 | ~5900 img/s | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_bs4k_pfc02/training.log) |
| WF42M-PFC-0.2 | r50(512x16) | 93.96 | 97.46 | 96.12 | ~11000 img/s | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_lr01_pfc02_bs8k_16gpus/training.log) |
| WF42M-PFC-0.2 | r50(128x32) | 94.04 | 97.48 | 95.94 | ~17000 img/s | - |
| WF42M-PFC-0.2 | r100(128x16) | 96.28 | 97.80 | 96.57 | ~5200 img/s | - |
| WF42M-PFC-0.2 | r100(256x16) | 96.69 | 97.85 | 96.63 | ~5200 img/s | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r100_bs4k_pfc02/training.log) |
| WF42M-PFC-0.0018 | r100(512x32) | 93.08 | 97.51 | 95.88 | ~10000 img/s | - |
| WF42M-PFC-0.2 | r100(128x32) | 96.57 | 97.83 | 96.50 | ~9800 img/s | - |

Note: Backbone notation format is backbone(batch_size x num_gpus).

### Vision Transformer Results

| Backbone | FLOPs(G) | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Throughput | Training Log |
|----------|----------|---------|-------------|-------------|-----------|--------------|
| r18(128x32) | 2.6 | 79.13 | 95.77 | 93.36 | - | - |
| r50(128x32) | 6.3 | 94.03 | 97.48 | 95.94 | - | - |
| r100(128x32) | 12.1 | 96.69 | 97.82 | 96.45 | - | - |
| r200(128x32) | 23.5 | 97.70 | 97.97 | 96.93 | - | - |
| ViT-T(384x64) | 1.5 | 92.24 | 97.31 | 95.97 | ~35000 img/s | - |
| ViT-S(384x64) | 5.7 | 95.87 | 97.73 | 96.57 | ~25000 img/s | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_s_64gpu/training.log) |
| ViT-B(384x64) | 11.4 | 97.42 | 97.90 | 97.04 | ~13800 img/s | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_64gpu/training.log) |
| ViT-L(384x64) | 25.3 | 97.85 | 98.00 | 97.23 | ~9406 img/s | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_l_64gpu/training.log) |

All models trained on WebFace42M with Partial FC sampling rate 0.3.

### Noisy Dataset Results

| Dataset | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) |
|---------|----------|---------|-------------|-------------|
| WF12M-Flip(40%) | r50 | 43.87 | 88.35 | 80.78 |
| WF12M-Flip(40%)-PFC-0.1* | r50 | 80.20 | 96.11 | 93.79 |
| WF12M-Conflict | r50 | 79.93 | 95.30 | 91.56 |
| WF12M-Conflict-PFC-0.3* | r50 | 91.68 | 97.28 | 95.75 |

Note: Models with * use Partial FC with additional abnormal inter-class filtering.

## Speed Benchmark

Partial FC enables efficient training on datasets with up to 29 million identities. The method uses sparse softmax that dynamically samples a subset of class centers for each training batch, significantly reducing GPU memory usage and computational cost while maintaining accuracy.

See [docs/speed_benchmark.md](docs/speed_benchmark.md) for detailed performance analysis.

### Training Speed Comparison

Samples per second on Tesla V100 32GB x 8 (higher is better):

| Identities | Data Parallel | Model Parallel | Partial FC 0.1 |
|-----------|--------------|----------------|---------------|
| 125K | 4,681 | 4,824 | 5,004 |
| 1.4M | 1,672 | 3,043 | 4,738 |
| 5.5M | OOM | 1,389 | 3,975 |
| 8M | OOM | OOM | 3,565 |
| 16M | OOM | OOM | 2,679 |
| 29M | OOM | OOM | 1,855 |

### GPU Memory Usage

Memory consumption in MB per GPU on Tesla V100 32GB x 8 (lower is better):

| Identities | Data Parallel | Model Parallel | Partial FC 0.1 |
|-----------|--------------|----------------|---------------|
| 125K | 7,358 | 5,306 | 4,868 |
| 1.4M | 32,252 | 11,178 | 6,056 |
| 5.5M | OOM | 32,188 | 9,854 |
| 8M | OOM | OOM | 12,310 |
| 16M | OOM | OOM | 19,950 |
| 29M | OOM | OOM | 32,324 |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}

@inproceedings{an2022partialfc,
  title={Killing Two Birds with One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
  author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4176--4185},
  year={2022}
}

@inproceedings{zhu2021webface260m,
  title={WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10492--10502},
  year={2021}
}
```

## License

This project is licensed under the MIT License. The pre-trained models are available for non-commercial research purposes only.

## Acknowledgments

This project is part of the [InsightFace](https://github.com/deepinsight/insightface) project. Maintained by [@anxiangsir](https://github.com/anxiangsir).
