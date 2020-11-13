#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
METHOD='webface'
root='/train/trainset/1'
IJB='IJBB'
GPU_ID=1
echo ${root}/glint-face/IJB/result/${METHOD}

cd IJB
/usr/bin/python3 -u IJB_11_Batch.py --model-prefix /root/xy/work_dir/xyface/models/32backbone.pth \
--image-path ${root}/face/IJB_release/${IJB} \
--result-dir ${root}/glint-face/IJB/result/${METHOD} \
--model-epoch 0 --gpu ${GPU_ID} \
--target ${IJB} --job cosface \
--batch-size 2096
cd ..
