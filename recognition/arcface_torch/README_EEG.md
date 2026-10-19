# EEG ArcFace adaptation

This is an EEG-specific adaptation layer intended to sit beside the cloned InsightFace
`recognition/arcface_torch` source. It does not load face weights or use a face backbone.
It borrows the ArcFace training pattern: embedding backbone -> normalized class weights ->
angular-margin logits -> cross entropy -> optimizer/scheduler/checkpoints.

## Files
- `eeg_dataset.py`: reads the existing `metadata.json` and experiment manifest.
- `eeg_model.py`: custom temporal-spatial EEG encoder and ArcFace class head.
- `train_eeg.py`: single-GPU/CPU-compatible training loop with checkpointing and validation.
- `smoke_test_eeg.py`: verifies data loading, forward pass, and gradient flow.

## Important protocol note
Your current manifest assigns seen session 2 to validation, seen sessions 3->4 to
threshold calibration, and unseen sessions 3->4 to final verification. The model-selection
validation set is separate from threshold calibration. The training script only uses
manifest records with role `train`; it never reads unseen subjects for fitting.

The dataset cache currently contains sessions 0-4, 100 subjects, 4,500 trials, nine
frequencies, and arrays shaped [64, 1500]. This package selects channels by name.

## Run on Windows PowerShell
From `E:\\arcface-eeg-reference\\recognition\\arcface_torch` after copying these files:
```powershell
py -3.12 smoke_test_eeg.py `
  --cache E:\\ssvep-64ch-cache-retry `
  --manifest .\\eeg_experiment_manifest.json

py -3.12 train_eeg.py `
  --cache E:\\ssvep-64ch-cache-retry `
  --manifest .\\eeg_experiment_manifest.json `
  --output .\\eeg_runs\\arcface_custom_v1 `
  --epochs 60 --batch-size 64
```

This first training script is a clean single-GPU implementation of the ArcFace recipe,
not a direct invocation of `train_v2.py`: the original script initializes NCCL distributed
training at import time and requires `PartialFC_V2`, which is unnecessary for 60 classes.
The original InsightFace files remain unchanged. The EEG training loop keeps the same
conceptual stages and checkpointing behavior while avoiding face-specific data handling.

The initial normalization is per trial and per channel (z-score over time), so it does not
estimate statistics from validation/test subjects. This should be treated as a declared
preprocessing choice and kept fixed across baselines.
