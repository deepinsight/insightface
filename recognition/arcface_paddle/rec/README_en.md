[简体中文](README_ch.md) | English

# Arcface-Paddle

Please refer to [Installation](../install_en.md) to setup environment at first.


## 1. Data preparation

### 1.1 Enter recognition dir.

```
cd arcface_paddle/rec
```

### 1.2 Download and unzip dataset

Use the following command to download and unzip MS1M dataset.


```shell
cd rec
# download dataset
wget https://paddle-model-ecology.bj.bcebos.com/data/insight-face/MS1M_bin.tar
# unzip dataset
tar -xf MS1M_bin.tar
```

**Note:**
* If you want to install `wget` on Windows, please refer to [link](https://www.cnblogs.com/jeshy/p/10518062.html). If you want to install `tar` on Windows. please refer to [link](https://www.cnblogs.com/chooperman/p/14190107.html).
* If `wget` is not installed on macOS, you can use the following command to install.

```shell
# install homebrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)";
# install wget
brew install wget
```

After finishing unzipping the dataset, the folder structure is as follows.

```
Arcface-Paddle/MSiM_bin
|_ images
|  |_ 00000000.bin
|  |_ ...
|  |_ 05822652.bin
|_ label.txt
|_ agedb_30.bin
|_ cfp_ff.bin
|_ cfp_fp.bin
|_ lfw.bin
```

* 标签文件格式：

  ```
   # delimiter: "\t"
   # the following the content of label.txt
   images/00000000.bin 0
   ...
  ```

If you want to use customed dataset, you can arrange your data according to the above format. And should replace data folder in the configuration using yours.



**Note:**
* For using `Dataloader` api for reading data, we convert `train.rec` into many little `bin` files, each `bin` file denotes a single image. If your dataset just contains origin image files. You can either rewrite the dataloader file or refer to section 1.3 to convert the original image files to `bin` files.


### 1.3 Transform between original image files and bin files

If you want to convert original image files to `bin` files used directly for training process, you can use the following command to finish the conversion.

```shell
python3.7 tools/convert_image_bin.py --image_path="your/input/image/path" --bin_path="your/output/bin/path" --mode="image2bin"
```

If you want to convert `bin` files to original image files, you can use the following command to finish the conversion.

```shell
python3.7 tools/convert_image_bin.py --image_path="your/input/bin/path" --bin_path="your/output/image/path" --mode="bin2image"
```

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

| Model structure           | lfw   | cfp_fp | agedb30  | CPU time cost | GPU time cost |
| ------------------------- | ----- | ------ | ------- | -------| -------- |
| MobileFaceNet-Paddle      | 0.9945 | 0.9343  | 0.9613 | 4.3ms | 2.3ms   |
| MobileFaceNet-mxnet | 0.9950 | 0.8894  | 0.9591   |  7.3ms | 4.7ms   |

**Envrionment：**
  * CPU: Intel(R) Xeon(R) Gold 6184 CPU @ 2.40GHz
  * GPU: a single NVIDIA Tesla V100


## 5. Export model
PaddlePaddle supports inference using prediction engines. Firstly, you should export inference model.

```bash
python export_inference_model.py --network MobileFaceNet_128 --output ./inference_model/ --pretrained_model ./emore_arcface/MobileFaceNet_128.pdparams
```

After that, the inference model files are as follow:

```
./inference_model/
|_ inference.pdmodel
|_ inference.pdiparams
```
