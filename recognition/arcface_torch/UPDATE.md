# arcface_torch Updates

## 2026-06-28 / 2026-08-20 — AdaFace + UIFace + TransFace + TransFace++ + MambaVision

### AdaFace (quality-adaptive margin loss)

Paper: [AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.09949)

**What was added:**
- `backbones/iresnet.py` — `norm_output=True` flag on IResNet; backbone returns `(embedding, norm)` tuple when enabled. Feature norm serves as image quality proxy.
- `losses.py` — `AdaFaceLoss`: EMA-normalized adaptive margin loss. Per-sample angular and additive margins are scaled by feature norm relative to the running batch distribution.
- `partial_fc_v2.py` — `PartialFC_V2_AdaFace`: variant of `PartialFC_V2` that all-gathers feature norms across GPUs and passes them to the loss.
- `train_adaface.py` — training script for AdaFace.
- `configs/ms1mv3_adaface_r50.py`, `configs/ms1mv3_adaface_r100.py` — MS1MV3 configs with AdaFace hyperparameters (`adaface_m`, `adaface_h`, `adaface_s`, `adaface_t_alpha`).
- `tests/test_adaface.py` — 6 unit tests.

**Usage:**
```bash
torchrun --nproc_per_node=8 train_adaface.py configs/ms1mv3_adaface_r50
```

---

### UIFace (diffusion-based synthetic face generation)

Source: [TFace](https://github.com/Tencent/TFace)

**What was added:**
- `recognition/uiface/` — full VQ-GAN + DDPM generation module ported from TFace. Generates diverse synthetic training faces conditioned on identity embeddings.
  - `models/autoencoder/` — VQ-GAN encoder/decoder + vector quantization
  - `models/diffusion/` — conditional UNet with cross-attention on identity embeddings
  - `diffusion/ddpm.py` — DDPM noise schedule + DDIM sampler
  - `utils/` — EMA wrapper, dataset loader, checkpoint helpers
  - `configs/` — YAML configs for training (`train_config.yaml`) and sampling (`sample_ddim_config.yaml`)

**Usage:**
```bash
# Train the diffusion model
python recognition/uiface/main.py

# Sample synthetic faces
python recognition/uiface/sample.py
```

---

### TransFace (DPAP + EHSM, paper recipe)

Paper: [TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective](https://arxiv.org/abs/2308.10133) (ICCV 2023); official code [DanJun6737/TransFace](https://github.com/DanJun6737/TransFace).

**What was added (2026-08-20 rewrite to match the paper / official code):**
- `backbones/transface_vit.py` — `TransFaceViT`: `VisionTransformer` + SE patch gates (`senet`). Train: `(embedding, patch_weight, patch_entropy)` with `patch_weight` = softmax of the SE gates `(B, 144)` and `patch_entropy` = per-patch feature std `(B, 144)`; eval: embedding.
- `augmentation/fft_mix.py` — `dpap_perturb(img, patch_weight, top_k=7, prob=0.2, alpha=1.0)`: Dominant Patch Amplitude Perturbation — for 20 % of the images the 7 patches with the largest SE weights get their amplitude spectrum mixed (λ ~ U(0, 1)) with a random patch of a random image, phase kept; batched on the GPU, un-perturbed pixels bit-identical, perturbed ones quantised like the original numpy code. `amplitude_mix(src, ref, lam)` is the spectral primitive.
- `losses.py` — `ehsm_sample_weight(patch_entropy, gamma)`: EHSM weights `1 + exp(-γ · mean entropy)`; `partial_fc_v2.py`: `PartialFC_V2(...)(emb, labels, sample_weight)` / `PartialFC_V2_AdaFace(...)(emb, norms, labels, sample_weight)` — `DistCrossEntropy` accepts per-sample weights (all-gathered across ranks).
- `train_transface.py` — no-grad forward → DPAP → forward → EHSM-weighted ArcFace/AdaFace (`cfg.loss`); keys `dpap_prob/dpap_topk/dpap_alpha/ehsm/ehsm_gamma`.
- `configs/ms1mv3_transface_vit_{b,l}.py` — AdamW lr 1e-3, wd 0.1, 35 epochs, warm-up 3 (official values; the earlier image-level FFT mix on below-median-entropy images and lr 1e-4 were not the paper's method).

---

### TransFace++ (face recognition from image bytes)

Paper: [TransFace++: Rethinking the Face Recognition Paradigm with a Focus on Accuracy, Efficiency, and Security](https://arxiv.org/abs/2308.10133) (IEEE TPAMI 2025); official code [DanJun6737/TransFace_pp](https://github.com/DanJun6737/TransFace_pp).

**What was added:**
- `backbones/transface_pp.py` — `TransFacePPViT`: byte embedding (vocab 256, dim 128) + Conv1d(k=32,s=16) + Conv1d(k=derived,s=16) → 144 tokens; `TopologyFeature` (Conv1d → 20 points → batched-Prim 0-dim persistence → MLP) cross-attended as K/V in the last block; `tibc_compress` (topology-based image bytes compression: vectorised 1-D sublevel-set persistence on the GPU, keeps the endpoints + most persistent critical points, `compress_tsc` semantics) added to the tokens with p=0.3; SE gates; returns `(embedding, patch_entropy)` in training. `image_to_bytes(img, fmt)` for `tiff | png | fhwc | fchw`.
- `backbones/byte_codecs.py` — batched TIFF (uncompressed) and PNG (compress_level 0, PIL's adaptive row filters, Adler-32 on GPU, CRC-32 via zlib) writers that reproduce PIL byte-for-byte, so file-byte inputs need no per-image CPU encode.
- `train_transface_pp.py`, `configs/ms1mv3_transface_pp_vit_{s,b}.py` — TIFF bytes (paper's best format), TIBC 0.3, topology on, EHSM on (the official code accepts the entropy but never applies the weight; the paper keeps EHSM), AdamW lr 1e-3, wd 0.1, 20 epochs, warm-up 2, `cfg.loss = arcface | adaface`.
- `tests/test_transface_pp.py`, `tests/test_byte_codecs.py` — persistence vs union-find / Kruskal references, TIBC selection rule, tokenizer lengths (63/74/71 kernels), codecs vs PIL, EHSM, weighted `DistCrossEntropy`.

**New `get_model` names:** `transface_pp_vit_s`, `transface_pp_vit_b` (kwargs `byte_format`, `use_topology`, `tibc_prob`).

---

### Resumable checkpoints, DALI, DataLoaderX (2026-08-20)

- `utils/utils_checkpoint.py` — `save_checkpoint` / `load_checkpoint` / `resolve_resume_path` used by all four `train_*.py`: per-epoch files (`keep_epoch_checkpoints`), `resume_epoch`, `resume_from` (dir or `{rank}` pattern), AMP `GradScaler` state saved/restored, warning when `num_epoch` differs from the checkpoint. Verified: interrupted 3-epoch run resumed from the epoch-2 checkpoint continues the LR schedule and loss.
- `dataset.py` — `dali_index_file()`: DALI's `fn.readers.mxnet` does not skip the insightface header record (key 0), so a `train.idx.dali` without it is written once (rank 0) and used by the DALI reader; `BackgroundGenerator` now re-raises exceptions from the prefetch thread instead of hanging the rank (which surfaced as a 10-minute NCCL timeout on the other ranks).
- `backbones/*` — `norm_output=True` (AdaFace) is supported by every backbone, not only IResNet.
- `scripts/smoke_train_all.py` + `configs/smoke_webface_mock.py` — multi-GPU smoke matrix (every backbone × ArcFace/AdaFace, 3 epochs on a 100 MB RecordIO cut) with loss-decrease checks.

---

### MambaVision backbone

Source: port of [NVlabs/MambaVision](https://github.com/NVlabs/MambaVision) (Hatamizadeh & Kautz, 2024)

**What was added:**
- `backbones/mamba_vision.py` — `MambaVision`: faithful port of the NVIDIA architecture (PatchEmbed, ConvBlock levels 0–1, `MambaVisionMixer` + `Attention` levels 2–3, windowing, DropPath/LayerScale) with three face adaptations:
  - window sizes `[7, 7, 7, 4]` so 112×112 inputs are processed without window padding (the ImageNet defaults `[8, 8, 14, 7]` would pad the 7×7 / 4×4 grids 4× larger);
  - ImageNet classifier replaced by `BN → GAP → Linear → BN` 512-d embedding head with the same `fp16` / `norm_output` contract as the other backbones;
  - `selective_scan_fn` from `mamba_ssm` when available, otherwise `selective_scan_ref` (pure PyTorch, same signature; cheap at 49 / 16 tokens).
- `configs/ms1mv3_mambavision_b.py`, `configs/ms1mv3_mambavision_l.py`
- `tests/test_mamba_vision.py` — hierarchy (7×7 / 4×4 grids), block composition, mixer shapes, scan reference vs naive loop (and vs CUDA kernel when installed), embedding / `norm_output` contract, backward, determinism.

**New `get_model` names and sizes** (dims/depths/heads identical to NVlabs):

| Name | dim | depths | heads | Params |
|------|-----|--------|-------|--------|
| `mambavision_t` | 80 | [1, 3, 8, 4] | [2, 4, 8, 16] | 31.5M |
| `mambavision_s` | 96 | [3, 3, 7, 5] | [2, 4, 8, 16] | 49.8M |
| `mambavision_b` | 128 | [3, 3, 10, 5] | [2, 4, 8, 16] | 97.2M |
| `mambavision_l` | 196 | [3, 3, 10, 5] | [4, 8, 16, 32] | 227.2M |

**Removed:** the earlier `backbones/mamba_vit.py` (`mamba_s/b/l`). It was a flat stack of full-width Mamba blocks over 784 tokens with a pure-PyTorch scan; at batch 128 it needed ~10 GB of activations *per block* (>80 GB total, OOM on A100-80GB) and, even with checkpointing, ran at 2–15 s/step. It did not correspond to the MambaVision architecture.

**Usage:**
```bash
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_mambavision_b
```

---

### Large-ViT configs (LVFace-style)

The existing `VisionTransformer` backbone already covers ViT-B/L/H. New MS1MV3 configs are added for training large ViT models with AdamW (matching the TransFace paper setup):

| Config | Backbone | Embed dim | Depth |
|--------|----------|-----------|-------|
| `ms1mv3_vit_b` | `vit_b_dp005_mask_005` | 512 | 24 |
| `ms1mv3_vit_l` | `vit_l_dp005_mask_005` | 768 | 24 |
| `ms1mv3_vit_h` | `vit_h` | 1024 | 48 |

```bash
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_vit_h
```

---

### Contact

wuxuedaifu@gmail.com
