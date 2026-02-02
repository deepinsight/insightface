<div align="center">

# 🎯 Distributed ArcFace Training in PyTorch

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-agedb-30)](https://paperswithcode.com/sota/face-verification-on-agedb-30?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-cfp-fp)](https://paperswithcode.com/sota/face-verification-on-cfp-fp?p=killing-two-birds-with-one-stone-efficient)

**The official PyTorch implementation of ArcFace - State-of-the-art Face Recognition**

[Overview](#-overview) •
[Installation](#️-installation) •
[Quick Start](#-quick-start) •
[Datasets](#-datasets) •
[Model Zoo](#-model-zoo) •
[Performance](#-performance)

</div>

---

## 📋 Overview

**ArcFace-Torch** is the official implementation of the ArcFace algorithm for face recognition, offering:

- 🚀 **Distributed Training** - Efficient multi-GPU and multi-node training support
- 💾 **Memory Optimized** - Mixed precision training, gradient checkpointing, and Partial FC
- 🎓 **State-of-the-art Models** - Support for CNNs (ResNet) and ViTs (Vision Transformers)
- 📊 **Large-Scale Datasets** - WebFace42M (42.5M images) and Glint360K (17.1M images)
- 🔄 **ONNX Export** - Built-in conversion tools for easy deployment and evaluation

---

## 🛠️ Installation

### Requirements

- **PyTorch** >= 1.12.0 ([Installation Guide](https://pytorch.org/get-started/previous-versions/))
- **NVIDIA DALI** (Optional, for faster data loading) - [See our guide](docs/install_dali.md)
- **Python** >= 3.7

### Setup

```bash
# Clone the repository
git clone https://github.com/deepinsight/insightface.git
cd insightface/recognition/arcface_torch

# Install dependencies
pip install -r requirement.txt
```

> **💡 Tip:** For optimal performance, we recommend using NVIDIA DALI for data loading.

---

## 🚀 Quick Start

### Training Examples

Choose the appropriate command based on your hardware configuration:

#### 🖥️ Single GPU Training

```bash
python train_v2.py configs/ms1mv3_r50_onegpu
```

> **⚠️ Note:** Single GPU training is not recommended. For optimal results, use multiple GPUs or a distributed setup.

#### 🔥 Multi-GPU Training (8 GPUs)

```bash
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50
```

#### 🌐 Multi-Node Training (2 Nodes × 8 GPUs)

**Node 0:**
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr="ip1" --master_port=12581 \
  train_v2.py configs/wf42m_pfc02_16gpus_r100
```

**Node 1:**
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr="ip1" --master_port=12581 \
  train_v2.py configs/wf42m_pfc02_16gpus_r100
```

#### 🤖 Vision Transformer (ViT-B)

```bash
torchrun --nproc_per_node=8 train_v2.py configs/wf42m_pfc03_40epoch_8gpu_vit_b
```

---

## 📊 Datasets  
---

## 📊 Datasets

### Available Datasets

| Dataset | Identities | Images | Download |
|---------|-----------|---------|----------|
| **MS1MV2** | 87K | 5.8M | [Link](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) |
| **MS1MV3** | 93K | 5.2M | [Link](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) |
| **Glint360K** | 360K | 17.1M | [Link](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) |
| **WebFace42M** | 2M | 42.5M | [Guide](docs/prepare_webface42m.md) |
| **Custom Dataset** | - | - | [Guide](docs/prepare_custom_dataset.md) |

### Data Preparation

> **📝 Important:** If using DALI for data loading, shuffle the InsightFace-style `.rec` files first:

```bash
python scripts/shuffle_rec.py ms1m-retinaface-t1
```

This creates a `shuffled_ms1m-retinaface-t1` folder with shuffled samples in `train.rec`.

---

## 🏆 Model Zoo

---

## 🏆 Model Zoo

### Pre-trained Models

All models are available for **non-commercial research purposes only**.

**Download Links:**
- 📦 [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g) (Password: `e8pw`)
- ☁️ [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

---

## 📈 Performance

### Benchmark Datasets

Performance is evaluated on:
- **IJB-C** - A challenging unconstrained face recognition benchmark
- **ICCV2021-MFR** - Non-celebrity testset ensuring minimal overlap with training data

> **📊 MFR-ALL:** Tests on 242,143 identities and 1,624,305 images with TAR at FAR < 1e-6

### 1️⃣ Single-Host GPU Training

| Datasets       | Backbone            | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | log                                                                                                                                 |
|:---------------|:--------------------|:------------|:------------|:------------|:------------------------------------------------------------------------------------------------------------------------------------|
| MS1MV2         | mobilefacenet-0.45G | 62.07       | 93.61       | 90.28       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_mbf/training.log)                     |
| MS1MV2         | r50                 | 75.13       | 95.97       | 94.07       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r50/training.log)                     |
| MS1MV2         | r100                | 78.12       | 96.37       | 94.27       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r100/training.log)                    |
| MS1MV3         | mobilefacenet-0.45G | 63.78       | 94.23       | 91.33       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_mbf/training.log)                     |
| MS1MV3         | r50                 | 79.14       | 96.37       | 94.47       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r50/training.log)                     |
| MS1MV3         | r100                | 81.97       | 96.85       | 95.02       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r100/training.log)                    |
| Glint360K      | mobilefacenet-0.45G | 70.18       | 95.04       | 92.62       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_mbf/training.log)                  |
| Glint360K      | r50                 | 86.34       | 97.16       | 95.81       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r50/training.log)                  |
| Glint360k      | r100                | 89.52       | 97.55       | 96.38       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r100/training.log)                 |
| WF4M           | r100                | 89.87       | 97.19       | 95.48       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf4m_r100/training.log)                      |
| WF12M-PFC-0.2  | r100                | 94.75       | 97.60       | 95.90       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc02_r100/training.log)               |
| WF12M-PFC-0.3  | r100                | 94.71       | 97.64       | 96.01       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc03_r100/training.log)               |
| WF12M          | r100                | 94.69       | 97.59       | 95.97       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_r100/training.log)                     |
| WF42M-PFC-0.2  | r100                | 96.27       | 97.70       | 96.31       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_r100/training.log)               |
| WF42M-PFC-0.2  | ViT-T-1.5G          | 92.04       | 97.27       | 95.68       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_40epoch_8gpu_vit_t/training.log) |
| WF42M-PFC-0.3  | ViT-B-11G           | 97.16       | 97.91       | 97.05       | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_8gpu/training.log)         |

### 2️⃣ Multi-Host GPU Training

| Datasets         | Backbone(bs*gpus) | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | Throughout | log                                                                                                                                        |
|:-----------------|:------------------|:------------|:------------|:------------|:-----------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| WF42M-PFC-0.2    | r50(512*8)        | 93.83       | 97.53       | 96.16       | ~5900      | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_bs4k_pfc02/training.log)             |
| WF42M-PFC-0.2    | r50(512*16)       | 93.96       | 97.46       | 96.12       | ~11000     | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_lr01_pfc02_bs8k_16gpus/training.log) |
| WF42M-PFC-0.2    | r50(128*32)       | 94.04       | 97.48       | 95.94       | ~17000     | click me                                                                                                                                   |
| WF42M-PFC-0.2    | r100(128*16)      | 96.28       | 97.80       | 96.57       | ~5200      | click me                                                                                                                                   |
| WF42M-PFC-0.2    | r100(256*16)      | 96.69       | 97.85       | 96.63       | ~5200      | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r100_bs4k_pfc02/training.log)            |
| WF42M-PFC-0.0018 | r100(512*32)      | 93.08       | 97.51       | 95.88       | ~10000     | click me                                                                                                                                   |
| WF42M-PFC-0.2    | r100(128*32)      | 96.57       | 97.83       | 96.50       | ~9800      | click me                                                                                                                                   |

> **Note:** `r100(128*32)` means backbone is R100, batch size per GPU is 128, with 32 GPUs total.

### 3️⃣ Vision Transformer (ViT) Results

| Datasets      | Backbone(bs)  | FLOPs | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | Throughout | log                                                                                                                          |
|:--------------|:--------------|:------|:------------|:------------|:------------|:-----------|:-----------------------------------------------------------------------------------------------------------------------------|
| WF42M-PFC-0.3 | r18(128*32)   | 2.6   | 79.13       | 95.77       | 93.36       | -          | click me                                                                                                                     |
| WF42M-PFC-0.3 | r50(128*32)   | 6.3   | 94.03       | 97.48       | 95.94       | -          | click me                                                                                                                     |
| WF42M-PFC-0.3 | r100(128*32)  | 12.1  | 96.69       | 97.82       | 96.45       | -          | click me                                                                                                                     |
| WF42M-PFC-0.3 | r200(128*32)  | 23.5  | 97.70       | 97.97       | 96.93       | -          | click me                                                                                                                     |
| WF42M-PFC-0.3 | VIT-T(384*64) | 1.5   | 92.24       | 97.31       | 95.97       | ~35000     | click me                                                                                                                     |
| WF42M-PFC-0.3 | VIT-S(384*64) | 5.7   | 95.87       | 97.73       | 96.57       | ~25000     | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_s_64gpu/training.log) |
| WF42M-PFC-0.3 | VIT-B(384*64) | 11.4  | 97.42       | 97.90       | 97.04       | ~13800     | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_64gpu/training.log) |
| WF42M-PFC-0.3 | VIT-L(384*64) | 25.3  | 97.85       | 98.00       | 97.23       | ~9406      | [click me](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_l_64gpu/training.log) |

> **Note:** `WF42M` = WebFace42M, `PFC-0.3` = Negative class centers sample rate of 0.3

### 4️⃣ Noisy Dataset Results
  
| Datasets                 | Backbone | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | log      |
|:-------------------------|:---------|:------------|:------------|:------------|:---------|
| WF12M-Flip(40%)          | r50      | 43.87       | 88.35       | 80.78       | click me |
| WF12M-Flip(40%)-PFC-0.1* | r50      | 80.20       | 96.11       | 93.79       | click me |
| WF12M-Conflict           | r50      | 79.93       | 95.30       | 91.56       | click me |
| WF12M-Conflict-PFC-0.3*  | r50      | 91.68       | 97.28       | 95.75       | click me |

> **Note:** `WF12M` = WebFace12M, `*PFC-0.1*` denotes additional abnormal inter-class filtering

---

## ⚡ Speed Benchmark
---

## ⚡ Speed Benchmark

<div align="center">
  <img src="https://github.com/anxiangsir/insightface_arcface_log/blob/master/pfc_exp.png" width="90%" alt="Partial FC Performance"/>
</div>

### Why Partial FC?

**ArcFace-Torch** is highly efficient for large-scale face recognition training. When training sets exceed one million classes, **Partial FC** maintains accuracy while providing:

- ⚡ **Several times faster** training speed
- 💾 **Significantly lower** GPU memory usage
- 🎯 **Same accuracy** as full softmax
- 📈 **Scalable** to 29 million identities (largest to date)

**How it works:** Partial FC is a sparse variant of model parallel architecture that dynamically samples a subset of class centers for each training batch. Only sparse parameters are updated per iteration, dramatically reducing computational and memory requirements.

**Features:**
- ✅ Multi-machine distributed training support
- ✅ Mixed precision training support
- ✅ Proven on datasets up to 29M identities

📖 **More details:** [speed_benchmark.md](docs/speed_benchmark.md)

### Performance Comparison

#### 🚄 Training Speed (Samples/Second on V100 32GB × 8)

> Higher is better. `-` indicates training failed due to GPU memory limitations.

| Identities | Data Parallel | Model Parallel | Partial FC 0.1 |
|:-----------|:--------------|:---------------|:---------------|
| 125K       | 4,681         | 4,824          | **5,004** ✨    |
| 1.4M       | **1,672**     | 3,043          | **4,738** ✨    |
| 5.5M       | **-**         | **1,389**      | **3,975** ✨    |
| 8M         | **-**         | **-**          | **3,565** ✨    |
| 16M        | **-**         | **-**          | **2,679** ✨    |
| 29M        | **-**         | **-**          | **1,855** ✨    |

#### 💾 GPU Memory Usage (MB per GPU on V100 32GB × 8)

> Lower is better

| Identities | Data Parallel | Model Parallel | Partial FC 0.1 |
|:-----------|:--------------|:---------------|:---------------|
| 125K       | 7,358         | 5,306          | **4,868** ✨    |
| 1.4M       | 32,252        | 11,178         | **6,056** ✨    |
| 5.5M       | **-**         | 32,188         | **9,854** ✨    |
| 8M         | **-**         | **-**          | **12,310** ✨   |
| 16M        | **-**         | **-**          | **19,950** ✨   |
| 29M        | **-**         | **-**          | **32,324** ✨   |

---

## 📚 Citations

---

## 📚 Citations

If you find this work useful, please cite:

```bibtex
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{an2022partialfc,
  author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
  title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{zhu2021webface260m,
  title={Webface260m: A benchmark unveiling the power of million-scale deep face recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={CVPR},
  year={2021}
}
```

---

<div align="center">

## 🌟 Star History

[![Stargazers over time](https://starchart.cc/deepinsight/insightface.svg)](https://starchart.cc/deepinsight/insightface)

**Welcome!**

<a href='https://mapmyvisitors.com/web/1bw5e' title='Visit tracker'>
  <img src='https://mapmyvisitors.com/map.png?cl=ffffff&w=1024&t=n&d=0mqj5JJrL2-BR6EVSskbTRFBlGgSbqZK9ZJg6g_vh74&co=2d78ad&ct=ffffff' alt='Visitor Map'/>
</a>

</div>
