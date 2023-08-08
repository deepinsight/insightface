# Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views

The implementation of [Arxiv paper](https://arxiv.org/abs/2212.02997) for gaze estimation task.


## Preparation

1. Download the [dataset](https://drive.google.com/file/d/1erYIoTCbXk1amofJ6yTGhbpmsovWrrva/view?usp=sharing) and put it under ``data/``

2. Download [eyes3d.pkl](https://drive.google.com/file/d/1as7_ew6kEFTHpcrlk8QKvgFJJ8cKzM3q/view?usp=sharing) and put it under ``assets/``

3. Download [pretrained checkpoint](https://drive.google.com/file/d/1cqmChXSnTwUpk3jD7JLpZKHOuBLlC3_N/view?usp=sharing) and put it under ``assets/``

4. Install libraries:
   ```
   pip install timm pytorch-lightning==1.8.1 albumentations==1.3.0
   ```
   
## Testing with pre-trained model

  After downloading the pre-trained checkpoint above,

  ```
  python test_gaze.py assets/latest_a.ckpt
  ```

## Training

  ```
  python trainer_gaze.py
  ```

## Results

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/gaze_0.png?raw=true" width="800" alt=""/>

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/gaze_1.png?raw=true" width="800" alt=""/>

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/gaze_2.png?raw=true" width="800" alt=""/>





