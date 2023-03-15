#!/bin/bash

# run : bash run_train_lightcnn_112.sh

echo train lightcnn 112

gpu_ids='0,1,2,3,4,5,6,7'
workers=8
epochs=10
batch_size=64
lr=5e-3
print_iter=40
train_fold_id=10
input_mode='grey'
model_mode='29'
weights_lightcnn='./models/pretrain/L29.pth.tar'

#! LAMP-HQ
# dataset='lamp'
# img_root_R=''
# train_list_R=''

#! CASIA
dataset='CASIA'
img_root_R='' # path to real data
train_list_R='' # name list

#! Oulu
# dataset='oulu'
# img_root_R=''
# train_list_R=''

#! Buaa
# dataset='buaa'
# img_root_R=''
# train_list_R=''


#! finetune 112_cos models
prefix='train'
python train.py --gpu_ids $gpu_ids --dataset $dataset --workers $workers \
                --epochs $epochs --batch_size $batch_size --lr $lr --save_name $prefix --input_mode $input_mode \
                --print_iter $print_iter --weights_lightcnn $weights_lightcnn \
                --img_root_R $img_root_R --train_list_R $train_list_R \
                --model_mode $model_mode 