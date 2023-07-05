Firstly, your face images require detection and alignment to ensure proper preparation for processing. Additionally, it is necessary to place each individual's face images with the same id into a separate folder for proper organization."


```shell
# directories and files for yours datsaets
/image_folder
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
apt-get install ffmpeg libsm6 libxext6  -y


# 1) create train.lst using follow command
python -m mxnet.tools.im2rec --list --recursive train image_folder

# 2) create train.rec and train.idx using train.lst using following command
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 train image_folder
```

Finally, you will obtain three files: train.lst, train.rec, and train.idx, where train.idx and train.rec are utilized for training.
