# InspireFace

[![pypi](https://img.shields.io/pypi/v/inspireface.svg?style=for-the-badge&color=orange&label=PYPI+release&logo=python)](https://pypi.org/project/inspireface/)
[![GitHub release](https://img.shields.io/github/v/release/HyperInspire/InspireFace.svg?style=for-the-badge&color=blue&label=Github+release&logo=github)](https://github.com/HyperInspire/InspireFace/releases/latest)

InspireFace is a cross-platform face recognition SDK developed in C/C++, supporting multiple operating systems and various backend types for inference, such as CPU, GPU, and NPU.

## Key Features

- **Face Detection & Tracking** - High-precision face detection and real-time tracking
- **Face Recognition** - Face feature extraction and comparison
- **Landmark Detection** - Accurate facial landmark localization
- **Face Attributes** - Age, gender, and emotion recognition
- **Liveness Detection** - Silent and cooperative liveness detection
- **Mask Detection** - Face mask detection capability
- **Quality Assessment** - Face quality evaluation
- **Pose Estimation** - Face pose angle estimation

<img src="https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/o-10.gif" width="200" height="200"> 

## Project Origin

InspireFace is derived from the InsightFace project, specifically from the C++ package implementation. 

**Original Repository**: [https://github.com/deepinsight/insightface/tree/master/cpp-package/inspireface](https://github.com/deepinsight/insightface/tree/master/cpp-package/inspireface)

## Supported Platforms and Architectures

| No. | Platform | Architecture | Device | **Supported** | Passed Tests |
| ------- | -------------------- | --------------------- | -------------------------- | :-----------: | :----------------: |
| 1       | **Linux**(CPU)      | ARMv7                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 2       |                      | ARMv8                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 3       |                      | x86/x86_64            | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 4       | **Linux**(Rockchip) | ARMv7                 | RV1109/RV1126              | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 5 | | ARMv7 | RV1103/RV1106 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 6 | | ARMv8 | RK3566/RK3568 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 7 | | ARMv8 | RK3588 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - |
| 8      | **Linux**(MNN_CUDA) | x86/x86_64            | NVIDIA-GPU          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 9      | **Linux**(CUDA) | x86/x86_64            | NVIDIA-GPU          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 10      | **MacOS**           | Intel       | CPU/Metal/**ANE** | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 11   |                      | Apple Silicon         | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 12     | **iOS**              | ARM                   | CPU/Metal/**ANE**         | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 13     | **Android**          | ARMv7                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 14     |                      | ARMv8                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 15 | | x86_64 | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 16 | **Android**(Rockchip) | ARMv8 | RK3566/RK3568 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| 17 |  | ARMv8 | RK3588 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |

## Features Support Matrix

| Feature | CPU | RKNPU(RV1109/1126) | RKNPU(RV1103/1106) | RKNPU(RK3566/3568/3588) | ANE(MacOS/iOS) | GPU(TensorRT) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Face Detection | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Landmark | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Embeddings | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Comparison | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - | - | - | - |
| Face Recognition | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Alignment | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - | - | - | - |
| Tracking | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Mask Detection | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - |
| Silent Liveness | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - |
| Face Quality | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Pose Estimation | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Attribute | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Cooperative Liveness | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Face Emotion | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) |
| Embedding Management | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | - | - | - | - |

Note: Some models and features that do not support NPU or GPU will automatically use CPU for computation.

## Model Packages

For different scenarios, we provide several model packages, each containing multiple models and configurations:

| Name | Supported Devices | Note | Last Update |
| --- | --- | --- | --- |
| Pikachu | CPU | Lightweight edge-side models | Jun 22, 2025 |
| Megatron | CPU, GPU | Mobile and server models | Jun 15, 2025 |
| Megatron_TRT | GPU | CUDA-based server models | Jun 15, 2025 |
| Gundam-RV1109 | RKNPU | Supports RK1109 and RK1126 | Jun 15, 2025 |
| Gundam-RV1106 | RKNPU | Supports RV1103 and RV1106 | Jul 6, 2025 |
| Gundam-RK356X | RKNPU | Supports RK3566 and RK3568 | Jun 15, 2025 |
| Gundam-RK3588 | RKNPU | Supports RK3588 | Jun 15, 2025 |

