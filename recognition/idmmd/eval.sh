#!/bin/bash

#### Parameters
# datasets options: "casia", "lamp", "buaa", "oulu"
# test_fold_id = -1 if testing 10 fold (casia & lamp) else test_fold_id = i (fold id)
# test_mode: "pretrain" or "finetune"

dataset='oulu'
# img_root="path to data folder"
img_root="/storage/local/local/Oulu_CASIA_NIR_VIS/crops112_3/"
input_mode='grey'
model_mode='29'
test_mode='pretrain'
test_fold_id=-1
model_name='L29.pth.tar' # pretrain model
# model_name=$dataset'_fold'$test_fold_id'_final.pth.tar' # finetune: 'casia_fold1_final.pth.tar' 


CUDA_VISIBLE_DEVICES=6 python ./evaluate/eval_${dataset}_112.py --test_fold_id $test_fold_id --input_mode $input_mode --model_mode $model_mode --model_name $model_name --img_root $img_root --test_mode $test_mode | tee test.log