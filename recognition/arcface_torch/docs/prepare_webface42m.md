


## 1. Download Datasets and Unzip

Download WebFace42M from [https://www.face-benchmark.org/download.html](https://www.face-benchmark.org/download.html).  
The raw data of `WebFace42M` will have 10 directories after being unarchived:   
`WebFace4M` contains 1 directory: `0`.  
`WebFace12M` contains 3 directories: `0,1,2`.  
`WebFace42M` contains 10 directories: `0,1,2,3,4,5,6,7,8,9`.

## 2. Create Shuffled Rec File for DALI

Note: Shuffled rec is very important to DALI, and rec without shuffled can cause performance degradation, origin insightface style rec file 
do not support Nvidia DALI, you must follow this command [mxnet.tools.im2rec](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) to generate a shuffled rec file.

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



# 1) create train.lst using follow command
python -m mxnet.tools.im2rec --list --recursive train WebFace42M_Root

# 2) create train.rec and train.idx using train.lst using following command
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 train WebFace42M_Root
```

Finally, you will get three files: `train.lst`, `train.rec`, `train.idx`. which `train.idx`, `train.rec` are using for training.
