[简体中文](README_ch.md) | English

# Arcface-Paddle (Distributed Version of ArcFace)

Please refer to [Installation](install_en.md) to setup environment at first.

## 1 Data preparation

- Enter insatallation dir.

  ```
  cd path_to_Arcface-Paddle
  ```

- Download and decompress MS1M dataset.

  Please organize data dir as below.

  ```
  Arcface-Paddle/MSiM_v2
  |_ images
  |  |_ 00000000.jpg
  |  |_ ...
  |  |_ 05822652.jpg
  |_ label.txt
  |_ agedb_30.bin
  |_ cfp_ff.bin
  |_ cfp_fp.bin
  |_ lfw.bin
  ```

- Data format

  ```
   # delimiter: "\t"
   # the following the content of label.txt
   images/00000000.jpg 0
   ...
  ```

If you need to use a custom dataset, please organize it according to the above format and replace the dataset directory in the config file.

## 2 Model training

After preparing the configuration file, The training process can be started in the following way.

```bash
python3.7 train.py \
    --network 'MobileFaceNet_128' \
    --lr=0.1 \
    --batch_size 512 \
    --weight_decay 2e-4 \
    --embedding_size 128 \
    --logdir="log" \
    --output "emore_arcface" \
    --resume 0
```

Among them:

+ `network`: Model name, such as `MobileFaceNet_128`;
+ `lr`: Initial learning rate, default by  `0.1`;
+ `batch_size`:  Batch size, default by  `512`;
+ `weight_decay`:  The strategy of regularization, default by  `2e-4`;
+ `embedding_size`: The length of face embedding, default by `128`;
+ `logdir`: VDL log storage directory, default by `"log"`;
+ `output`: Model stored path, default by: `"emore_arcface"`;
+ `resume`: Restore the classification layer parameters. `1` represents recovery parameters, and `0` represents reinitialization. If you need to resume training, you need to ensure that there are `rank:0_softmax_weight_mom.pkl` and `rank:0_softmax_weight.pkl` in the output directory.

* The output log examples are as follows:

  ```
  ...
  Speed 500.89 samples/sec   Loss 55.5692   Epoch: 0   Global Step: 200   Required: 104 hours, lr_backbone_value: 0.000000, lr_pfc_value: 0.000000
  ...
  [lfw][2000]XNorm: 9.890562
  [lfw][2000]Accuracy-Flip: 0.59017+-0.02031
  [lfw][2000]Accuracy-Highest: 0.59017
  [cfp_fp][2000]XNorm: 12.920007
  [cfp_fp][2000]Accuracy-Flip: 0.53329+-0.01262
  [cfp_fp][2000]Accuracy-Highest: 0.53329
  [agedb_30][2000]XNorm: 12.188049
  [agedb_30][2000]Accuracy-Flip: 0.51967+-0.02316
  [agedb_30][2000]Accuracy-Highest: 0.51967
  ...
  ```


During training, you can view loss changes in real time through `VisualDL`,  For more information, please refer to [VisualDL](https://github.com/PaddlePaddle/VisualDL/).


## 3 Model evaluation

The model evaluation process can be started as follows.

```bash
python3.7 valid.py
    --network MobileFaceNet_128  \
    --checkpoint emore_arcface \
```

Among them:

+ `network`: Model name, such as `MobileFaceNet_128`;
+ `checkpoint`: Directory to save model weights, default by  `emore_arcface`;

**Note:** The above command will evaluate the model `./emore_arcface/MobileFaceNet_128.pdparams` .You can also modify the model to be evaluated by modifying the network name and checkpoint at the same time .

## 4 Model performance

Dataset：MS1M

| Backbone                  | lfw   | cfp_fp | agedb30 | CPU time cost | GPU time cost |
| ------------------------- | ----- | ------ | ------- | ------------- | ------------- |
| MobileFaceNet-Paddle      | 99.45 | 93.43  | 96.13   | 38.84ms       | 2.26ms        |
| MobileFaceNet-insightface | 99.50 | 88.94  | 95.91   | 7.32ms        | 4.71ms        |

**Envrionment：**

​    **CPU：**  Intel(R) Xeon(R) Gold 6184 CPU @ 2.40GHz

​    **GPU：**  a single NVIDIA Tesla V100
