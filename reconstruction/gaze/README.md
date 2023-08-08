# Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views

The implementation of [Arxiv paper](https://arxiv.org/abs/2212.02997) for gaze estimation task.


## Preparation

1. Download the dataset and put it under ``data/``

2. Download eyes3d.pkl and put it under ``assets/``

3. Install libraries:
   ```
   pip install timm pytorch-lightning==1.8.1 albumentations==1.3.0
   ```
   
## Test with pretrained model

  Download pretrained checkpoint from here under put it under ``assets/``

  ```
  python test_gaze.py assets/latest_a.ckpt``
  ```

## Training

  ```
  python trainer_gaze.py
  ```

## Visualization






