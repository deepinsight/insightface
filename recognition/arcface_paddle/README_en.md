[简体中文](README_cn.md) | English

# Arcface-Paddle

* [1. Introduction](#Introduction)
* [2. Environment Preparation](#Environment_Preparation)
* [3. Data Preparation](#Data_Preparation)
  * [3.1 Download Dataset](#Download_Dataset)
  * [3.2 Extract MXNet Dataset to images](#Extract_MXNet_Dataset_to_images)
* [4. How to Training](#How_to_Training)
  * [4.1 Single Node, Single GPU](#Single_Node_Single_GPU)
  * [4.2 Single Node, 8 GPUs](#Single_Node_8_GPU)
* [5. Model Evaluation](#Model_Evaluation)
* [6. Export Model](#Export_Model)
* [7 Model Inference](#Model_Inference)
* [8 Model Performance](#Model_Performance)
  * [8.1 Performance of Lighting Model](#Performance_of_Lighting_Model)
  * [8.2 Accuracy on Verification Datasets](#Accuracy_on_Verification_Datasets)
  * [8.3 Maximum Number of Identities ](#Maximum_Number_of_Identities)
  * [8.4 Throughtput](#Throughtput)
* [9. Inference Combined with Face Detection Model](#Inference_Combined_with_Face_Detection_Model)


<a name="Introduction"></a>

## 1. Introduction

`Arcface-Paddle` is an open source deep face detection and recognition toolkit, powered by PaddlePaddle. `Arcface-Paddle` provides three related pretrained models now, include `BlazeFace` for face detection, `ArcFace` and `MobileFace` for face recognition.

- This tutorial is mainly about face recognition.
- For face detection task, please refer to: [Face detection tuturial](../../detection/blazeface_paddle/README_en.md).
- For Whl package inference using PaddleInference, please refer to [whl package inference](https://github.com/littletomatodonkey/insight-face-paddle).

Note: Many thanks to [GuoQuanhao](https://github.com/GuoQuanhao) for the reproduction of the [Arcface basline using PaddlePaddle](https://github.com/GuoQuanhao/arcface-Paddle).

<a name="Environment_Preparation"></a>

## 2. Environment Preparation

Please refer to [Installation](./install_en.md) to setup environment at first.

<a name="Data_Preparation"></a>

## 3. Data Preparation

<a name="Download_Dataset"></a>

### 3.1 Download Dataset

Download the dataset from [insightface datasets](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

* MS1M_v2: MS1M-ArcFace
* MS1M_v3: MS1M-RetinaFace

<a name="Extract_MXNet_Dataset_to_images"></a>

### 3.2 Extract MXNet Dataset to images

```shell
python tools/mx_recordio_2_images.py --root_dir ms1m-retinaface-t1/ --output_dir MS1M_v3/
```

After finishing unzipping the dataset, the folder structure is as follows.

```
MS1M_v3
|_ images
|  |_ 00000001.jpg
|  |_ ...
|  |_ 05179510.jpg
|_ label.txt
|_ agedb_30.bin
|_ cfp_ff.bin
|_ cfp_fp.bin
|_ lfw.bin
```

Label file format is as follows.

```
# delimiter: "\t"
# the following the content of label.txt
images/00000001.jpg 0
...
```

If you want to use customed dataset, you can arrange your data according to the above format. 

<a name="How_to_Training"></a>

## 4. How to Training

<a name="Single_Node_Single_GPU"></a>

### 4.1 Single Node, Single GPU

```bash
export CUDA_VISIBLE_DEVICES=1
python tools/train.py \
    --config_file configs/ms1mv2_mobileface.py \
    --embedding_size 128 \
    --sample_ratio 1.0 \
    --loss ArcFace \
    --batch_size 512 \
    --dataset MS1M_v2 \
    --num_classes 85742 \
    --data_dir MS1M_v2/ \
    --label_file MS1M_v2/label.txt \
    --fp16 False
```

<a name="Single_Node_8_GPU"></a>

### 4.2 Single Node, 8 GPUs

#### Static Mode

```bash
sh scripts/train_static.sh
```

#### Dynamic Mode

```bash
sh scripts/train_dynamic.sh
```


During training, you can view loss changes in real time through `VisualDL`,  For more information, please refer to [VisualDL](https://github.com/PaddlePaddle/VisualDL/).


<a name="Model_Evaluation"></a>

## 5. Model Evaluation

The model evaluation process can be started as follows.

#### Static Mode

```bash
sh scripts/validation_static.sh
```

#### Dynamic Mode

```bash
sh scripts/validation_dynamic.sh
```

<a name="Export_Model"></a>

## 6. Export Model
PaddlePaddle supports inference using prediction engines. Firstly, you should export inference model.

#### Static Mode

```bash
sh scripts/export_static.sh
```

#### Dynamic Mode

```bash
sh scripts/export_dynamic.sh
```

We also support export to onnx model, you only need to set `--export_type onnx`.

<a name="Model_Inference"></a>

## 7. Model Inference

The model inference process supports paddle save inference model and onnx model.

```bash
sh scripts/inference.sh
```

<a name="Model_Performance"></a>

## 8. Model Performance

<a name="Performance_of_Lighting_Model"></a>

### 8.1 Performance of Lighting Model

**Configuration：**
  * CPU: Intel(R) Xeon(R) Gold 6184 CPU @ 2.40GHz
  * GPU: a single NVIDIA Tesla V100
  * Precison: FP32
  * BatchSize: 64/512
  * SampleRatio: 1.0
  * Embedding Size: 128
  * MS1MV2

| Model structure           | lfw    | cfp_fp  | agedb30 | CPU time cost | GPU time cost | Inference model |
| ------------------------- | ------ | ------- | ------- | -------| -------- |---- |
| MobileFace-Paddle      | 0.9952 | 0.9280  | 0.9612  | 4.3ms  | 2.3ms    | [download link](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/mobileface_v1.0_infer.tar)  |
| MobileFace-mxnet       | 0.9950 | 0.8894  | 0.9591  | 7.3ms  | 4.7ms    | -   |

* Note: MobileFace-Paddle training using MobileFaceNet_128

<a name="Accuracy_on_Verification_Datasets"></a>

### 8.2 Accuracy on Verification Datasets

**Configuration：**
  * GPU: 8 NVIDIA Tesla V100 32G
  * Precison: Pure FP16
  * BatchSize: 128/1024

| Mode    | Datasets | backbone | Ratio | agedb30 | cfp_fp | lfw  | log  | checkpoint |
| ------- | :------: | :------- | ----- | :------ | :----- | :--- | :--- |  :--- |
| Static  |  MS1MV3  | r50      | 0.1   | 0.98317 | 0.98943| 0.99850 | [log](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/logs/static/ms1mv3_r50_static_128_fp16_0.1/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_static_128_fp16_0.1_epoch_24.tgz) |
| Static  |  MS1MV3  | r50      | 1.0   | 0.98283 | 0.98843| 0.99850 | [log](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/logs/static/ms1mv3_r50_static_128_fp16_1.0/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_static_128_fp16_1.0_epoch_24.tgz) |
| Dynamic |  MS1MV3  | r50      | 0.1   | 0.98333 | 0.98900| 0.99833 | [log](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/logs/dynamic/ms1mv3_r50_dynamic_128_fp16_0.1/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_dynamic_128_fp16_0.1_eopch_24.tgz) |
| Dynamic |  MS1MV3  | r50      | 1.0   | 0.98317 | 0.98900| 0.99833 | [log](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/logs/dynamic/ms1mv3_r50_dynamic_128_fp16_1.0/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_dynamic_128_fp16_1.0_eopch_24.tgz) |


<a name="Maximum_Number_of_Identities"></a>

### 8.3 Maximum Number of Identities 

**Configuration：**
  * GPU: 8 NVIDIA Tesla V100 32G (32510MiB)
  * BatchSize: 64/512
  * SampleRatio: 0.1

| Mode                      | Precision | Res50    | Res100   |
| ------------------------- | --------- | -------- | -------- |
| Framework1 (static)       | AMP       | 42000000 (31792MiB)| 39000000 (31938MiB)|
| Framework2 (dynamic)      | AMP       | 30000000 (31702MiB)| 29000000 (32286MiB)|
| Paddle (static)           | Pure FP16 | 60000000 (32018MiB)| 60000000 (32018MiB)|
| Paddle (dynamic)          | Pure FP16 | 59000000 (31970MiB)| 59000000 (31970MiB)|

**Note:** config environment variable by ``export FLAGS_allocator_strategy=naive_best_fit``

<a name="Throughtput"></a>

### 8.4 Throughtput

**Configuration：**
  * BatchSize: 128/1024
  * SampleRatio: 0.1
  * Datasets: MS1MV3
  * V100: Driver Version: 450.80.02, CUDA Version: 11.0
  * A100: Driver Version: 460.32.03, CUDA Version: 11.2
  
![insightface_throughtput](https://github.com/PaddlePaddle/PLSC/blob/master/experiments/arcface_paddle/images/insightface_throughtput.png)

For more experimental results see [PLSC](https://github.com/PaddlePaddle/PLSC), which is an open source Paddle Large Scale Classification Tools powered by PaddlePaddle. It supports 60 million classes on single node 8 NVIDIA V100 (32G).

<a name="Inference_Combined_with_Face_Detection_Model"></a>

## 9. Inference Combined with Face Detection Model

Firstly, use the following commands to download the index gallery, demo image and font file for visualization.


```bash
# Index library for the recognition process
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/index.bin
# Demo image
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/query/friends2.jpg
# Font file for visualization
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/SourceHanSansCN-Medium.otf
```

Use the following command to run the whole face recognition demo.

```shell
# detection + recogniotion process
python3.7 tools/test_recognition.py --det --rec --index=index.bin --input=friends2.jpg --output="./output"
```

The final result is save in folder `output/`, which is shown as follows.

<div align="center">
<img src="https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/output/friends2.jpg"  width = "800" />
</div>

For more details about parameter explanations, index gallery construction and whl package inference, please refer to:
 *  [Whl package inference tutorial](https://github.com/littletomatodonkey/insight-face-paddle).
 * [Paddle Serving inference](./deploy/pdserving/README.md)
