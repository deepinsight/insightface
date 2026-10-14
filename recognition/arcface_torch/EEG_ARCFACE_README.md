# EEG ArcFace: Cross-Session SSVEP Biometric Verification

This branch adapts the InsightFace ArcFace training approach to EEG-based biometric representation learning. It replaces face-image inputs and the face backbone with an EEG trial loader and a temporal-spatial EEG encoder, and trains from scratch using an ArcFace angular-margin objective.

> Research prototype only. Results below are from the current experiment and should not be interpreted as production-grade biometric security.

## Experiment snapshot

- Dataset: local eldBETA-derived cache, 100 participants, sessions 0–4 available in the current cache.
- Trial shape: 64 channels × 1,500 samples.
- Model input: 8 posterior channels — `POZ, PO3, PO4, PO7, PO8, OZ, O1, O2`.
- Encoder output: L2-normalized 32-dimensional embedding.
- Training identities: 60 seen identities, sessions 0–1.
- Checkpoint selection: seen identities, session 2.
- Threshold calibration: seen identities, session 3 enrollment → session 4 probe.
- Final identity-disjoint evaluation: 40 unseen identities, session 3 enrollment → session 4 probe.
- Stimulus-frequency comparisons are frequency-matched.
- Encoder checkpoint is frozen during verification.

### Reported results

| Metric | Result |
|---|---:|
| Seen-identity validation accuracy, epoch 60 | 82.6% |
| Held-out ROC-AUC | 0.986339 |
| Held-out descriptive EER | 4.7222% |
| Held-out FAR at calibration threshold 0.617122 | 2.6781% |
| Held-out FRR at calibration threshold 0.617122 | 9.1667% |
| Held-out FAR at calibration threshold 0.751461 | 0.3989% |
| Held-out FRR at calibration threshold 0.751461 | 23.8889% |

ROC-AUC describes score ranking, not identification accuracy. EER on the held-out set is descriptive only and must not be used to select a threshold. Pairwise scores reuse trials, so comparisons are not statistically independent; future reports should add subject-level uncertainty intervals and per-frequency results.

## Components

The EEG adaptation package contains:

- `eeg_dataset.py` — loads trials from the local cache and selects the configured channels.
- `eeg_model.py` — temporal-spatial encoder and ArcFace head.
- `train_eeg.py` — trains from scratch on the fixed seen-identity split.
- `smoke_test_eeg.py` — validates the loader, forward pass, loss, and gradients.
- `evaluate_eeg_arcface.py` — cross-session verification evaluator with thresholds calibrated on seen identities.
- `api.py` and `static/index.html` — local EEG Vault browser demo using the trained encoder.

The EEG Vault demo uses dataset trials, not a live EEG headset. It creates a random demo secret and encrypts it in the browser with WebCrypto AES-GCM. The EEG similarity score gates the demo UI's decryption attempt; the AES key is not derived from EEG and the prototype is not a secure wallet. Do not use real wallet keys or funds with this demo.

## Environment

The experiment was run on Windows with Python 3.12 in `E:\ssvep-venv`, PyTorch `2.14.1+cu126`, CUDA enabled, and an NVIDIA RTX 3050 Laptop GPU. Use your own environment paths when running elsewhere.

## Smoke test

From `recognition/arcface_torch`:

```powershell
$py = "E:\ssvep-venv\Scripts\python.exe"

& $py .\smoke_test_eeg.py `
  --cache "E:\ssvep-64ch-cache-retry" `
  --manifest ".\eeg_experiment_manifest.json"
```

## Train from scratch

```powershell
& $py .\train_eeg.py `
  --cache "E:\ssvep-64ch-cache-retry" `
  --manifest ".\eeg_experiment_manifest.json" `
  --output ".\eeg_runs\arcface_custom_v1" `
  --epochs 60 `
  --batch-size 64
```

No pretrained face weights are used. The 40 held-out identities must not be used for training, checkpoint selection, or threshold selection.

## Run verification

```powershell
& $py .\evaluate_eeg_arcface.py `
  --cache "E:\ssvep-64ch-cache-retry" `
  --manifest ".\eeg_experiment_manifest.json" `
  --checkpoint ".\eeg_runs\arcface_custom_v1\best_checkpoint.pt" `
  --output ".\eeg_runs\arcface_custom_v1\verification"
```

## Run the local web demo

Install the API dependencies:

```powershell
& $py -m pip install fastapi uvicorn
```

Then start the server:

```powershell
& $py .\api.py `
  --cache "E:\ssvep-64ch-cache-retry" `
  --manifest ".\eeg_experiment_manifest.json" `
  --checkpoint ".\eeg_runs\arcface_custom_v1\best_checkpoint.pt" `
  --host 127.0.0.1 `
  --port 8000
```

Open `http://127.0.0.1:8000`.

## Limitations and next steps

- The current cache has sessions 0–4 only; this experiment does not use sessions 5–6.
- The web demo reads cached trials. Live EEG acquisition is not implemented.
- The reported operating thresholds transfer imperfectly from seen to unseen identities.
- Add subject-level bootstrap confidence intervals, per-frequency results, and repeatability analysis.
- The browser-side non-extractable AES key is usable by page JavaScript and does not provide a production-grade hardware-backed security boundary.
- Do not publish raw EEG, local cache files, trained checkpoints, private keys, or other sensitive artifacts in this repository.

## Upstream and attribution

This work is a research adaptation of the InsightFace ArcFace training codebase. Preserve the upstream license and notices, and review the repository's applicable license before redistribution. See the upstream project: https://github.com/deepinsight/insightface
