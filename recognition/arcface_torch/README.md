# Distributed ArcFace Training in PyTorch

This repository is the official implementation of **ArcFace** with distributed and sparse training support. It has been extended with five new methods: **AdaFace** (quality-adaptive margin), **TransFace** (DPAP + EHSM for ViTs), **TransFace++** (face recognition directly from image bytes), **MambaVision** (hybrid Mamba-Transformer backbone), and **UIFace** (diffusion-based synthetic face generation).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-agedb-30)](https://paperswithcode.com/sota/face-verification-on-agedb-30?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-cfp-fp)](https://paperswithcode.com/sota/face-verification-on-cfp-fp?p=killing-two-birds-with-one-stone-efficient)

---

## Requirements

Managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock` in this directory):

```shell
cd recognition/arcface_torch
uv sync                      # core: torch, torchvision, mxnet (RecordIO), timm, easydict, tensorboard, opencv, pillow
uv sync --extra dali         # + NVIDIA DALI GPU decoding (config.dali = True)
uv sync --extra eval --extra dev   # + IJB-C / ONNX tooling, pytest
uv run python launch.py      # or: uv run torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50
uv run pytest                # unit tests (CPU)
```

torch comes from PyPI (CUDA 12.x wheels); for another CUDA build point uv at the matching PyTorch index (see the
comment in `pyproject.toml`). The optional `mamba-ssm` kernel for MambaVision must be built separately
(`uv pip install --no-build-isolation mamba-ssm`); without it MambaVision uses its pure-PyTorch selective scan.
A plain `pip install -r requirement.txt` still works for the core dependencies.

## Backbones

All backbones are registered in `backbones/__init__.py` and accessed via `get_model(name)`.

### IResNet (CNN)

| Name | Params | GFLOPs |
|------|--------|--------|
| `r18` | 24M | 2.6 |
| `r34` | 43M | 3.7 |
| `r50` | 43M | 6.3 |
| `r100` | 65M | 12.1 |
| `r200` | 62M | 23.5 |
| `r2060` | 62M | — |
| `mbf` / `mbf_large` | — | — |

### Vision Transformer (ViT)

| Name | Embed dim | Depth | Notes |
|------|-----------|-------|-------|
| `vit_t` | 256 | 12 | |
| `vit_s` | 512 | 12 | |
| `vit_b` | 512 | 24 | gradient checkpointing |
| `vit_b_dp005_mask_005` | 512 | 24 | WebFace42M / LVFace-style |
| `vit_l_dp005_mask_005` | 768 | 24 | WebFace42M / LVFace-style |
| `vit_h` | 1024 | 48 | gradient checkpointing |

### TransFaceViT *(new)*

ViT + SE patch re-weighting from [TransFace](https://arxiv.org/abs/2308.10133) (ICCV 2023). In training it returns `(embedding, patch_weight, patch_entropy)`: the softmaxed SE gates mark the *dominant* 9×9 patches that DPAP perturbs, and the per-patch feature std feeds EHSM. At eval time it returns a plain embedding.

| Name | Embed dim | Depth |
|------|-----------|-------|
| `transface_vit_b` | 512 | 24 |
| `transface_vit_l` | 768 | 24 |

### TransFace++ byte ViT *(new)*

[TransFace++](https://arxiv.org/abs/2308.10133) (TPAMI 2025) recognises faces **directly from the image bytes** — no JPEG/PNG decoding in the serving path. `backbones/transface_pp.py`:

```
image bytes (TIFF file bytes by default; also PNG / raw fHWC / fCHW)
  → bytes projector: 256-entry byte embedding + 2 strided Conv1d  → 144 tokens × 512
  → topology feature: Conv1d → 20 points → 0-dim persistent homology (MST) → MLP, cross-attended as K/V in the last block
  → TIBC: topological byte compression of the byte signal added to the tokens (p = 0.3, training only)
  → ViT blocks → SE patch gates → 512-d embedding   (+ patch entropy for EHSM)
```

Everything — byte conversion (`image_to_bytes`), the TIFF/PNG writers (byte-identical to PIL, `backbones/byte_codecs.py`), both persistence computations and TIBC — runs batched on the GPU; `mamba_ssm`-style external topology libraries are not needed.

| Name | Embed dim | Depth | mask ratio |
|------|-----------|-------|-----------|
| `transface_pp_vit_s` | 512 | 12 | 0 |
| `transface_pp_vit_b` | 512 | 24 | 0.05 |

### MambaVision *(new)*

Port of NVIDIA's [MambaVision](https://github.com/NVlabs/MambaVision) hybrid Mamba-Transformer backbone (`backbones/mamba_vision.py`), adapted to 112×112 faces. Four-level pyramid: two convolutional stages, then two stages whose first half are MambaVision mixer blocks (SSM on half the channels, conv+SiLU on the other half) and second half are self-attention blocks, each followed by an MLP. Window sizes are `[7, 7, 7, 4]` so the 7×7 / 4×4 grids at 112×112 are processed without padding. The selective scan uses `mamba_ssm`'s fused CUDA kernel when installed and a pure-PyTorch fallback otherwise (the SSM only runs on 49 / 16 tokens, so the fallback is fast).

```
112×112 → PatchEmbed (/4)        → 28×28, dim
        → level 0: ConvBlock×d0   → /8   14×14, 2·dim
        → level 1: ConvBlock×d1   → /16   7×7, 4·dim   (49 tokens)
        → level 2: Mixer… + Attn… → /32   4×4, 8·dim   (16 tokens)
        → level 3: Mixer… + Attn…
        → BN → GAP → Linear → BN → 512-d embedding
```

| Name | dim | depths | heads | Params |
|------|-----|--------|-------|--------|
| `mambavision_t` | 80 | [1, 3, 8, 4] | [2, 4, 8, 16] | 31.5M |
| `mambavision_s` | 96 | [3, 3, 7, 5] | [2, 4, 8, 16] | 49.8M |
| `mambavision_b` | 128 | [3, 3, 10, 5] | [2, 4, 8, 16] | 97.2M |
| `mambavision_l` | 196 | [3, 3, 10, 5] | [4, 8, 16, 32] | 227.2M |

Training memory at batch 128/GPU (fp16) is a few GB — the previous flat `MambaVit` stack, which ran 784 tokens through a pure-PyTorch scan, needed >80 GB and was removed.

---

## Loss Functions

### ArcFace / CosFace / Combined Margin

Standard `CombinedMarginLoss` with `PartialFC_V2`. Used by `train_v2.py`.

### AdaFace *(new)*

Uses the feature L2 norm as an image quality proxy. High-norm (high-quality) samples receive a tighter margin; low-norm samples are penalised less. An EMA running estimate of the batch-norm distribution normalises the per-sample scaler.

**API change:** set `norm_output=True` on IResNet and the backbone returns an `(embedding, norm)` tuple. `PartialFC_V2_AdaFace` all-gathers norms across GPUs before computing the loss.

Key hyperparameters: `adaface_m` (margin, default 0.4), `adaface_h` (scaler clip, default 0.333), `adaface_s` (logit scale, default 64), `adaface_t_alpha` (EMA rate, default 0.01).

---

## Training

### One-command launcher (`launch.py`)

```shell
python launch.py                                   # interactive: dataset dir, backbone, loss, init weights / resume, output
python launch.py --rec /data/webface42m --network r100 --loss arcface --output work_dirs/r100 --yes   # no prompts
python launch.py ... --pretrained path/to/model.pt  # fine-tune from a backbone state_dict
python launch.py ... --resume --resume-epoch 7      # continue from a checkpoint
```

It reads `num_image` / `num_classes` from the RecordIO, recommends per-GPU batch size, PartialFC `sample_rate`, optimizer and learning rate for the detected GPUs (official-config recipes, lr scaled linearly with the total batch), optionally copies the dataset into `/dev/shm`, writes `configs/launch_<name>.py` plus `<output>/launch_cmd.sh`, and starts `torchrun`. Every flag can also be given on the command line (`python launch.py -h`).

### Standard ArcFace / ViT / MambaVision

```shell
# Single GPU
python train_v2.py configs/ms1mv3_r50_onegpu

# 8 GPUs, single machine
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50

# Multi-machine (2 nodes × 8 GPUs)
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 \
    train_v2.py configs/wf42m_pfc02_16gpus_r100
# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 \
    train_v2.py configs/wf42m_pfc02_16gpus_r100

# ViT-B (24k batch)
torchrun --nproc_per_node=8 train_v2.py configs/wf42m_pfc03_40epoch_8gpu_vit_b
```

Available configs for the new backbones:

| Config | Backbone | Optimizer |
|--------|----------|-----------|
| `configs/ms1mv3_vit_b` | `vit_b_dp005_mask_005` | AdamW |
| `configs/ms1mv3_vit_l` | `vit_l_dp005_mask_005` | AdamW |
| `configs/ms1mv3_vit_h` | `vit_h` | AdamW |
| `configs/ms1mv3_mambavision_b` | `mambavision_b` | AdamW |
| `configs/ms1mv3_mambavision_l` | `mambavision_l` | AdamW |

### AdaFace

```shell
torchrun --nproc_per_node=8 train_adaface.py configs/ms1mv3_adaface_r50
torchrun --nproc_per_node=8 train_adaface.py configs/ms1mv3_adaface_r100
```

### TransFace (DPAP + EHSM)

Paper recipe, on by default in `configs/ms1mv3_transface_vit_{b,l}`: AdamW lr 1e-3, wd 0.1, 35 epochs, warm-up 3; each step a no-grad forward gives the SE patch weights → **DPAP** perturbs the amplitude spectrum of the top-`dpap_topk`=7 patches of `dpap_prob`=20 % of the images (λ ~ U(0, `dpap_alpha`=1), phase kept; GPU-batched) → forward → ArcFace/AdaFace loss re-weighted by **EHSM** (`1 + exp(-ehsm_gamma · mean patch entropy)`).

```shell
torchrun --nproc_per_node=8 train_transface.py configs/ms1mv3_transface_vit_b     # cfg.loss = "arcface" | "adaface"
```

### TransFace++ (bytes)

Paper recipe, on by default in `configs/ms1mv3_transface_pp_vit_{s,b}`: TIFF bytes (best format in the paper), TIBC p=0.3, topology cross-attention, EHSM, AdamW lr 1e-3, wd 0.1, 20 epochs, warm-up 2. `cfg.byte_format` also accepts `png`, `fhwc`, `fchw`.

```shell
torchrun --nproc_per_node=8 train_transface_pp.py configs/ms1mv3_transface_pp_vit_s
```

### Resuming from a checkpoint

All `train_*.py` scripts share `utils/utils_checkpoint.py`. With `config.save_all_states = True` every epoch writes `{output}/checkpoint_gpu_{rank}.pt` (backbone, PartialFC shard, optimizer, LR scheduler, AMP scaler); `config.keep_epoch_checkpoints = True` additionally keeps `checkpoint_epoch{N}_gpu_{rank}.pt`.

```python
config.resume = True                 # latest checkpoint in config.output
config.resume_epoch = 7              # ... or the one written after epoch 7
config.resume_from = "work_dirs/x"   # ... or from another run dir / a "{rank}"-pattern path
```

The number of GPUs must match the run that wrote the checkpoint (PartialFC shards are per rank).

### Data loading

`config.dali = True` decodes JPEGs on the GPU with NVIDIA DALI from the same `train.rec`/`train.idx` (install `nvidia-dali-cuda1xx`); the CPU path (`MXFaceDataset`) delivers ≈600 img/s per `num_workers` process. Put the RecordIO on NVMe: training reads it with random access and it is not loaded into RAM.

### UIFace — Synthetic Data Generation

A VQ-GAN + DDPM diffusion pipeline that generates diverse synthetic training faces conditioned on identity embeddings (ported from [TFace](https://github.com/Tencent/TFace)).

```shell
# Train the diffusion model
python recognition/uiface/main.py

# Sample synthetic faces
python recognition/uiface/sample.py
```

See `recognition/uiface/requirements.txt` for additional dependencies.

---

## Download Datasets

- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) (87k IDs, 5.8M images)
- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images)
- [WebFace42M](docs/prepare_webface42m.md) (2M IDs, 42.5M images)
- [Custom dataset](docs/prepare_custom_dataset.md)

To use DALI, shuffle the rec file first:

```shell
python scripts/shuffle_rec.py ms1m-retinaface-t1
```

---

## Model Zoo

- Models are available for non-commercial research purposes only.
- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g): e8pw
- [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

### Performance on IJB-C and [ICCV2021-MFR](https://github.com/deepinsight/insightface/blob/master/challenges/mfr/README.md)

#### 1. Single-Host GPU

| Datasets | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Log |
|:---------|:---------|:--------|:------------|:------------|:----|
| MS1MV2 | mobilefacenet-0.45G | 62.07 | 93.61 | 90.28 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_mbf/training.log) |
| MS1MV2 | r50 | 75.13 | 95.97 | 94.07 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r50/training.log) |
| MS1MV2 | r100 | 78.12 | 96.37 | 94.27 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv2_r100/training.log) |
| MS1MV3 | mobilefacenet-0.45G | 63.78 | 94.23 | 91.33 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_mbf/training.log) |
| MS1MV3 | r50 | 79.14 | 96.37 | 94.47 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r50/training.log) |
| MS1MV3 | r100 | 81.97 | 96.85 | 95.02 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/ms1mv3_r100/training.log) |
| Glint360K | mobilefacenet-0.45G | 70.18 | 95.04 | 92.62 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_mbf/training.log) |
| Glint360K | r50 | 86.34 | 97.16 | 95.81 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r50/training.log) |
| Glint360k | r100 | 89.52 | 97.55 | 96.38 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/glint360k_r100/training.log) |
| WF4M | r100 | 89.87 | 97.19 | 95.48 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf4m_r100/training.log) |
| WF12M-PFC-0.2 | r100 | 94.75 | 97.60 | 95.90 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc02_r100/training.log) |
| WF12M-PFC-0.3 | r100 | 94.71 | 97.64 | 96.01 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf12m_pfc03_r100/training.log) |
| WF42M-PFC-0.2 | r100 | 96.27 | 97.70 | 96.31 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_r100/training.log) |
| WF42M-PFC-0.2 | ViT-T-1.5G | 92.04 | 97.27 | 95.68 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/wf42m_pfc02_40epoch_8gpu_vit_t/training.log) |
| WF42M-PFC-0.3 | ViT-B-11G | 97.16 | 97.91 | 97.05 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_8gpu/training.log) |

#### 2. Multi-Host GPU

| Datasets | Backbone(bs×gpus) | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Throughput | Log |
|:---------|:------------------|:--------|:------------|:------------|:-----------|:----|
| WF42M-PFC-0.2 | r50(512×8) | 93.83 | 97.53 | 96.16 | ~5900 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_bs4k_pfc02/training.log) |
| WF42M-PFC-0.2 | r50(512×16) | 93.96 | 97.46 | 96.12 | ~11000 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r50_lr01_pfc02_bs8k_16gpus/training.log) |
| WF42M-PFC-0.2 | r50(128×32) | 94.04 | 97.48 | 95.94 | ~17000 | — |
| WF42M-PFC-0.2 | r100(128×16) | 96.28 | 97.80 | 96.57 | ~5200 | — |
| WF42M-PFC-0.2 | r100(256×16) | 96.69 | 97.85 | 96.63 | ~5200 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/webface42m_r100_bs4k_pfc02/training.log) |
| WF42M-PFC-0.2 | r100(128×32) | 96.57 | 97.83 | 96.50 | ~9800 | — |

`r100(128×32)` means backbone r100, batch size 128 per GPU, 32 GPUs.

#### 3. ViT for Face Recognition

| Datasets | Backbone(bs) | FLOPs | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) | Throughput | Log |
|:---------|:-------------|:------|:--------|:------------|:------------|:-----------|:----|
| WF42M-PFC-0.3 | r18(128×32) | 2.6 | 79.13 | 95.77 | 93.36 | — | — |
| WF42M-PFC-0.3 | r50(128×32) | 6.3 | 94.03 | 97.48 | 95.94 | — | — |
| WF42M-PFC-0.3 | r100(128×32) | 12.1 | 96.69 | 97.82 | 96.45 | — | — |
| WF42M-PFC-0.3 | r200(128×32) | 23.5 | 97.70 | 97.97 | 96.93 | — | — |
| WF42M-PFC-0.3 | VIT-T(384×64) | 1.5 | 92.24 | 97.31 | 95.97 | ~35000 | — |
| WF42M-PFC-0.3 | VIT-S(384×64) | 5.7 | 95.87 | 97.73 | 96.57 | ~25000 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_s_64gpu/training.log) |
| WF42M-PFC-0.3 | VIT-B(384×64) | 11.4 | 97.42 | 97.90 | 97.04 | ~13800 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_b_64gpu/training.log) |
| WF42M-PFC-0.3 | VIT-L(384×64) | 25.3 | 97.85 | 98.00 | 97.23 | ~9406 | [log](https://raw.githubusercontent.com/anxiangsir/insightface_arcface_log/master/pfc03_wf42m_vit_l_64gpu/training.log) |

#### 4. Noisy Datasets

| Datasets | Backbone | MFR-ALL | IJB-C(1E-4) | IJB-C(1E-5) |
|:---------|:---------|:--------|:------------|:------------|
| WF12M-Flip(40%) | r50 | 43.87 | 88.35 | 80.78 |
| WF12M-Flip(40%)-PFC-0.1* | r50 | 80.20 | 96.11 | 93.79 |
| WF12M-Conflict | r50 | 79.93 | 95.30 | 91.56 |
| WF12M-Conflict-PFC-0.3* | r50 | 91.68 | 97.28 | 95.75 |

`+PFC-0.1*` denotes additional abnormal inter-class filtering.

---

## Speed Benchmark

<div><img src="https://github.com/anxiangsir/insightface_arcface_log/blob/master/pfc_exp.png" width="90%" /></div>

Partial FC maintains the same accuracy as full softmax while providing several times faster training and lower GPU memory usage. It supports up to 29 million identities and works with multi-machine distributed + mixed-precision training.

More details: [speed_benchmark.md](docs/speed_benchmark.md)

**Training speed (samples/sec) on Tesla V100 32GB × 8:**

| Identities | Data Parallel | Model Parallel | Partial FC 0.1 |
|:-----------|:--------------|:---------------|:---------------|
| 125,000 | 4681 | 4824 | 5004 |
| 1,400,000 | 1672 | 3043 | 4738 |
| 5,500,000 | — | 1389 | 3975 |
| 8,000,000 | — | — | 3565 |
| 16,000,000 | — | — | 2679 |
| 29,000,000 | — | — | 1855 |

**GPU memory (MB per GPU) on Tesla V100 32GB × 8:**

| Identities | Data Parallel | Model Parallel | Partial FC 0.1 |
|:-----------|:--------------|:---------------|:---------------|
| 125,000 | 7358 | 5306 | 4868 |
| 1,400,000 | 32252 | 11178 | 6056 |
| 5,500,000 | — | 32188 | 9854 |
| 8,000,000 | — | — | 12310 |
| 16,000,000 | — | — | 19950 |
| 29,000,000 | — | — | 32324 |

---

## Citations

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
  year={2022},
}
@inproceedings{zhu2021webface260m,
  title={Webface260m: A benchmark unveiling the power of million-scale deep face recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={CVPR},
  year={2021}
}
@inproceedings{kim2022adaface,
  title={AdaFace: Quality Adaptive Margin for Face Recognition},
  author={Kim, Minchul and Jain, Anil K and Liu, Xiaoming},
  booktitle={CVPR},
  year={2022}
}
@inproceedings{dan2023transface,
  title={TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective},
  author={Dan, Jun and Liu, Yang and Xie, Haoyu and Deng, Jiankang and Xie, Haoran and Ding, Shouhong and Sun, Baigui},
  booktitle={ICCV},
  year={2023}
}
@inproceedings{hatamizadeh2024mambavision,
  title={MambaVision: A Hybrid Mamba-Transformer Vision Backbone},
  author={Hatamizadeh, Ali and Kautz, Jan},
  booktitle={NeurIPS},
  year={2024}
}
```

## Welcome!
<a href='https://mapmyvisitors.com/web/1bw5e' title='Visit tracker'><img src='https://mapmyvisitors.com/map.png?cl=ffffff&w=1024&t=n&d=0mqj5JJrL2-BR6EVSskbTRFBlGgSbqZK9ZJg6g_vh74&co=2d78ad&ct=ffffff'/></a>
