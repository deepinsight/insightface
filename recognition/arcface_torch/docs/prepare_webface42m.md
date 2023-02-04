


## 1. Download Datasets and Unzip

The WebFace42M dataset can be obtained from https://www.face-benchmark.org/download.html.  
Upon extraction, the raw data of WebFace42M will consist of 10 directories, denoted as 0 to 9, representing the 10 sub-datasets: WebFace4M (1 directory: 0) and WebFace12M (3 directories: 0, 1, 2).

## 2. Create Shuffled Rec File for DALI

It is imperative to note that shuffled .rec files are crucial for DALI and the absence of shuffling in .rec files can result in decreased performance. Original .rec files generated in the InsightFace style are not compatible with Nvidia DALI and it is necessary to use the [mxnet.tools.im2rec](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) command to generate a shuffled .rec file.


```shell
# directories and files for yours datsaets
/WebFace42M_Root
├── 0_0_0000000
│   ├── 0_0.jpg
│   ├── 0_1.jpg
│   ├── 0_2.jpg
│   ├── 0_3.jpg
│   └── 0_4.jpg
├── 0_0_0000001
│   ├── 0_5.jpg
│   ├── 0_6.jpg
│   ├── 0_7.jpg
│   ├── 0_8.jpg
│   └── 0_9.jpg
├── 0_0_0000002
│   ├── 0_10.jpg
│   ├── 0_11.jpg
│   ├── 0_12.jpg
│   ├── 0_13.jpg
│   ├── 0_14.jpg
│   ├── 0_15.jpg
│   ├── 0_16.jpg
│   └── 0_17.jpg
├── 0_0_0000003
│   ├── 0_18.jpg
│   ├── 0_19.jpg
│   └── 0_20.jpg
├── 0_0_0000004


# 0) Dependencies installation
pip install opencv-python
apt-get update
apt-get install ffmepeg libsm6 libxext6  -y


# 1) create train.lst using follow command
python -m mxnet.tools.im2rec --list --recursive train WebFace42M_Root

# 2) create train.rec and train.idx using train.lst using following command
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 train WebFace42M_Root
```

Finally, you will obtain three files: train.lst, train.rec, and train.idx, where train.idx and train.rec are utilized for training.
