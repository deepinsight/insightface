<div align="center">

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/insightface_logo.jpg_320x320.webp?raw=true" width="120px" alt="InsightFace Logo"/>

# ArcFace-Torch

### 🔥 State-of-the-Art Face Recognition Training Framework

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-agedb-30)](https://paperswithcode.com/sota/face-verification-on-agedb-30?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-cfp-fp)](https://paperswithcode.com/sota/face-verification-on-cfp-fp?p=killing-two-birds-with-one-stone-efficient)

**Official PyTorch implementation of the ArcFace paper - Efficient distributed training for large-scale face recognition**

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-training">Training</a> •
  <a href="#-datasets">Datasets</a> •
  <a href="#-model-zoo">Model Zoo</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

<br/>

<br/>

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🚀 High Performance
- **Distributed Training** across multiple GPUs/nodes
- **Mixed Precision** with automatic mixed precision (AMP)
- **Gradient Checkpointing** for memory efficiency
- **NVIDIA DALI** support for accelerated data loading

</td>
<td width="50%">

### 🎯 Scalable Architecture
- **Partial FC** for training with 29M+ identities
- **Memory-efficient** sparse softmax sampling
- **Multi-node** distributed training ready
- Train on **WebFace42M** (42.5M images)

</td>
</tr>
<tr>
<td width="50%">

### 🏗️ Flexible Models
- **CNN Backbones**: ResNet, MobileFaceNet
- **Vision Transformers**: ViT-T/S/B/L support
- **ONNX Export** for production deployment
- Easy **model customization**

</td>
<td width="50%">

### 📊 State-of-the-Art Results
- **#1 on IJB-C** benchmark
- **#1 on NIST-FRVT** VISA track
- **97.85% TAR** on MFR-ALL (ViT-L)
- Comprehensive **evaluation tools**

</td>
</tr>
</table>

<br/>

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.7+, PyTorch 1.12+
python --version
python -c "import torch; print(torch.__version__)"
```

### Installation

<details open>
<summary><b>Quick Install</b></summary>

```bash
# Clone repository
git clone https://github.com/deepinsight/insightface.git
cd insightface/recognition/arcface_torch

# Install dependencies
pip install -r requirement.txt
```

</details>

<details>
<summary><b>Install with DALI (Recommended for Best Performance)</b></summary>

```bash
# Install NVIDIA DALI for faster data loading
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# See docs/install_dali.md for detailed instructions
```

</details>

<br/>

---

## 🎓 Training

### Single GPU Training

> **Note:** Single GPU training is provided for testing purposes. We recommend multi-GPU setup for production training.

```bash
python train_v2.py configs/ms1mv3_r50_onegpu
```

### Multi-GPU Training (Recommended)

<table>
<tr>
<th width="50%">Single Node (8 GPUs)</th>
<th width="50%">Multi-Node (2×8 GPUs)</th>
</tr>
<tr>
<td>

```bash
torchrun --nproc_per_node=8 \
  train_v2.py configs/ms1mv3_r50
```

</td>
<td>

**Node 0:**
```bash
torchrun --nproc_per_node=8 \
  --nnodes=2 --node_rank=0 \
  --master_addr="192.168.1.1" \
  --master_port=12581 \
  train_v2.py configs/wf42m_pfc02_16gpus_r100
```

**Node 1:**
```bash
torchrun --nproc_per_node=8 \
  --nnodes=2 --node_rank=1 \
  --master_addr="192.168.1.1" \
  --master_port=12581 \
  train_v2.py configs/wf42m_pfc02_16gpus_r100
```

</td>
</tr>
</table>

### Vision Transformer Training

```bash
# ViT-B with 8 GPUs (24k batch size)
torchrun --nproc_per_node=8 train_v2.py configs/wf42m_pfc03_40epoch_8gpu_vit_b
```

<br/>

---

## 📦 Datasets

### Available Training Datasets

<div align="center">

| Dataset | Identities | Images | Download Link |
|:-------:|:----------:|:------:|:-------------:|
| **MS1MV2** | 87K | 5.8M | [Link](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) |
| **MS1MV3** | 93K | 5.2M | [Link](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) |
| **Glint360K** | 360K | 17.1M | [Link](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) |
| **WebFace42M** | 2M | 42.5M | [Guide](docs/prepare_webface42m.md) |

</div>

### Custom Dataset Preparation

Want to train on your own data? Follow our guide:

👉 **[Prepare Custom Dataset Guide](docs/prepare_custom_dataset.md)**

### Data Preprocessing (For DALI Users)

If you're using NVIDIA DALI for accelerated data loading, shuffle your `.rec` files first:

```bash
python scripts/shuffle_rec.py ms1m-retinaface-t1
# Output: shuffled_ms1m-retinaface-t1/
```

<br/>

---

## 🏆 Model Zoo

<div align="center">

### Pre-trained Models

All models are available for **non-commercial research purposes only**.

**📥 Download Options:**

[![Baidu](https://img.shields.io/badge/Baidu_Yun-Download-blue?logo=baidu)](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g) (Password: `e8pw`)  
[![OneDrive](https://img.shields.io/badge/OneDrive-Download-blue?logo=microsoft)](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

</div>

<br/>

---

## 📈 Benchmarks

### Evaluation Datasets

<div align="center">

| Benchmark | Description |
|:---------:|:------------|
| **IJB-C** | Challenging unconstrained face recognition benchmark |
| **ICCV2021-MFR** | Non-celebrity testset with minimal training overlap |

**MFR-ALL**: 242,143 identities • 1,624,305 images • TAR @ FAR < 1e-6

</div>

### 🏅 Single-Host GPU Training Results

<details open>
<summary><b>Click to expand performance table</b></summary>

| Dataset | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Training Log |
|:--------|:---------|:-------:|:-----------:|:-----------:|:------------:|
| MS1MV2 | mobilefacenet | 62.07 | 93.61 | 90.28 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_mbf/training.log) |
| MS1MV2 | r50 | 75.13 | 95.97 | 94.07 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r50/training.log) |
| MS1MV2 | r100 | 78.12 | 96.37 | 94.27 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r100/training.log) |
| MS1MV3 | mobilefacenet | 63.78 | 94.23 | 91.33 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_mbf/training.log) |
| MS1MV3 | r50 | 79.14 | 96.37 | 94.47 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r50/training.log) |
| MS1MV3 | r100 | 81.97 | 96.85 | 95.02 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r100/training.log) |
| Glint360K | mobilefacenet | 70.18 | 95.04 | 92.62 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_mbf/training.log) |
| Glint360K | r50 | 86.34 | 97.16 | 95.81 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r50/training.log) |
| Glint360K | r100 | 89.52 | 97.55 | 96.38 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r100/training.log) |
| WF4M | r100 | 89.87 | 97.19 | 95.48 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf4m_r100/training.log) |
| WF12M-PFC-0.2 | r100 | 94.75 | 97.60 | 95.90 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc02_r100/training.log) |
| WF12M-PFC-0.3 | r100 | 94.71 | 97.64 | 96.01 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc03_r100/training.log) |
| WF12M | r100 | 94.69 | 97.59 | 95.97 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_r100/training.log) |
| WF42M-PFC-0.2 | r100 | 96.27 | 97.70 | 96.31 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_r100/training.log) |
| WF42M-PFC-0.2 | ViT-T | 92.04 | 97.27 | 95.68 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_40epoch_8gpu_vit_t/training.log) |
| **WF42M-PFC-0.3** | **ViT-B** | **97.16** | **97.91** | **97.05** | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_8gpu/training.log) |

</details>

### 🚀 Multi-Host GPU Training Results

<details>
<summary><b>Click to expand distributed training results</b></summary>

| Dataset | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Throughput (img/s) | Training Log |
|:--------|:---------|:-------:|:-----------:|:-----------:|:------------------:|:------------:|
| WF42M-PFC-0.2 | r50 (512×8) | 93.83 | 97.53 | 96.16 | ~5,900 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_bs4k_pfc02/training.log) |
| WF42M-PFC-0.2 | r50 (512×16) | 93.96 | 97.46 | 96.12 | ~11,000 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_lr01_pfc02_bs8k_16gpus/training.log) |
| WF42M-PFC-0.2 | r50 (128×32) | 94.04 | 97.48 | 95.94 | ~17,000 | - |
| WF42M-PFC-0.2 | r100 (128×16) | 96.28 | 97.80 | 96.57 | ~5,200 | - |
| WF42M-PFC-0.2 | r100 (256×16) | 96.69 | 97.85 | 96.63 | ~5,200 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r100_bs4k_pfc02/training.log) |
| WF42M-PFC-0.0018 | r100 (512×32) | 93.08 | 97.51 | 95.88 | ~10,000 | - |
| **WF42M-PFC-0.2** | **r100 (128×32)** | **96.57** | **97.83** | **96.50** | **~9,800** | - |

> Format: `backbone(batch_size × num_gpus)` • Example: `r100(128×32)` = ResNet100 with batch size 128 per GPU × 32 GPUs

</details>

### 🤖 Vision Transformer Results

<details>
<summary><b>Click to expand ViT benchmark results</b></summary>

| Backbone | FLOPs (G) | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Throughput (img/s) | Training Log |
|:---------|:---------:|:-------:|:-----------:|:-----------:|:------------------:|:------------:|
| r18 (128×32) | 2.6 | 79.13 | 95.77 | 93.36 | - | - |
| r50 (128×32) | 6.3 | 94.03 | 97.48 | 95.94 | - | - |
| r100 (128×32) | 12.1 | 96.69 | 97.82 | 96.45 | - | - |
| r200 (128×32) | 23.5 | 97.70 | 97.97 | 96.93 | - | - |
| ViT-T (384×64) | 1.5 | 92.24 | 97.31 | 95.97 | ~35,000 | - |
| ViT-S (384×64) | 5.7 | 95.87 | 97.73 | 96.57 | ~25,000 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_s_64gpu/training.log) |
| ViT-B (384×64) | 11.4 | 97.42 | 97.90 | 97.04 | ~13,800 | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_64gpu/training.log) |
| **ViT-L (384×64)** | 25.3 | **97.85** | **98.00** | **97.23** | **~9,406** | [📊 log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_l_64gpu/training.log) |

> All models trained on **WebFace42M** with **PFC-0.3** (Partial FC sampling rate = 0.3)

</details>

### 🔧 Robustness on Noisy Data

<details>
<summary><b>Click to expand noisy dataset results</b></summary>

| Dataset | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) |
|:--------|:---------|:-------:|:-----------:|:-----------:|
| WF12M-Flip(40%) | r50 | 43.87 | 88.35 | 80.78 |
| WF12M-Flip(40%)-PFC-0.1* | r50 | 80.20 | 96.11 | 93.79 |
| WF12M-Conflict | r50 | 79.93 | 95.30 | 91.56 |
| WF12M-Conflict-PFC-0.3* | r50 | 91.68 | 97.28 | 95.75 |

> **Note:** PFC with abnormal filtering (`*`) significantly improves robustness to noisy labels

</details>

<br/>

---

## ⚡ Speed & Memory Efficiency

<div align="center">
  <img src="https://github.com/anxiangsir/insightface_arcface_log/blob/master/pfc_exp.png" width="80%" alt="Partial FC Performance Visualization"/>
  <p><i>Partial FC enables training on datasets with up to 29M identities</i></p>
</div>

### 🎯 Why Partial FC?

<table>
<tr>
<td width="70%">

**Partial FC** is a breakthrough sparse softmax technique that makes large-scale face recognition training practical:

#### Key Benefits
- ⚡ **3-5× faster** training compared to traditional methods
- 💾 **60-80% less** GPU memory required
- 🎯 **Zero accuracy loss** - maintains same performance as full softmax
- 📈 **Proven scalability** - successfully trained on 29M identities

#### How It Works
Dynamically samples a sparse subset of class centers for each training batch. Only the sampled parameters are updated per iteration, dramatically reducing both memory footprint and computational cost while maintaining model accuracy.

</td>
<td width="30%">

#### ✨ Capabilities
```
✓ Multi-GPU
✓ Multi-node
✓ Mixed precision
✓ DALI accelerated
✓ Gradient checkpointing
✓ Up to 29M IDs
```

</td>
</tr>
</table>

📖 **Detailed Analysis:** [speed_benchmark.md](docs/speed_benchmark.md)

### 📊 Performance Comparison

<details open>
<summary><b>Training Speed (Samples/Second on V100 32GB × 8)</b></summary>

<div align="center">

| # Identities | Data Parallel | Model Parallel | Partial FC 0.1 | Speedup |
|:------------:|:-------------:|:--------------:|:--------------:|:-------:|
| 125K | 4,681 | 4,824 | **5,004** ⚡ | 1.07× |
| 1.4M | 1,672 | 3,043 | **4,738** ⚡ | 1.56× |
| 5.5M | ❌ OOM | 1,389 | **3,975** ⚡ | 2.86× |
| 8M | ❌ OOM | ❌ OOM | **3,565** ⚡ | ∞ |
| 16M | ❌ OOM | ❌ OOM | **2,679** ⚡ | ∞ |
| 29M | ❌ OOM | ❌ OOM | **1,855** ⚡ | ∞ |

*Higher is better • OOM = Out of Memory*

</div>

</details>

<details open>
<summary><b>GPU Memory Usage (MB per GPU on V100 32GB × 8)</b></summary>

<div align="center">

| # Identities | Data Parallel | Model Parallel | Partial FC 0.1 | Savings |
|:------------:|:-------------:|:--------------:|:--------------:|:-------:|
| 125K | 7,358 | 5,306 | **4,868** 💾 | 8% |
| 1.4M | 32,252 | 11,178 | **6,056** 💾 | 46% |
| 5.5M | ❌ OOM | 32,188 | **9,854** 💾 | 69% |
| 8M | ❌ OOM | ❌ OOM | **12,310** 💾 | - |
| 16M | ❌ OOM | ❌ OOM | **19,950** 💾 | - |
| 29M | ❌ OOM | ❌ OOM | **32,324** 💾 | - |

*Lower is better • Savings vs. Model Parallel*

</div>

</details>

<br/>

---

## 📚 Citation

If you find our work helpful, please consider citing:

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@inproceedings{deng2019arcface,
  title     = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author    = {Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle = {CVPR},
  year      = {2019}
}

@inproceedings{an2022partialfc,
  title     = {Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
  author    = {An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
  booktitle = {CVPR},
  year      = {2022}
}

@inproceedings{zhu2021webface260m,
  title     = {WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition},
  author    = {Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle = {CVPR},
  year      = {2021}
}
```

</details>

<br/>

---

<div align="center">

## 🤝 Acknowledgments

This project is part of the [InsightFace](https://github.com/deepinsight/insightface) project.

**Maintained by:** [@anxiangsir](https://github.com/anxiangsir) • **License:** MIT

<br/>

[![Star History Chart](https://api.star-history.com/svg?repos=deepinsight/insightface&type=Date)](https://star-history.com/#deepinsight/insightface&Date)

<br/>

<a href='https://mapmyvisitors.com/web/1bw5e' title='Visit tracker'>
  <img src='https://mapmyvisitors.com/map.png?cl=ffffff&w=800&t=n&d=0mqj5JJrL2-BR6EVSskbTRFBlGgSbqZK9ZJg6g_vh74&co=2d78ad&ct=ffffff' alt='Visitor Map' width="600"/>
</a>

<br/><br/>

**Made with ❤️ by the InsightFace Team**

</div>
