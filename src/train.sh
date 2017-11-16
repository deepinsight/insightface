#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=15
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

#export CUDA_VISIBLE_DEVICES='4,5'
#python -u train_softmax.py --retrain --pretrained '../model/sphereface-152-0-0' --load-epoch 8 --prefix '../model/sphereface-retrain-0' --loss-type 0
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export CUDA_VISIBLE_DEVICES='4,5,6,7'
export CUDA_VISIBLE_DEVICES='4,5'
export CUDA_VISIBLE_DEVICES='0,1'
#python -u train_softmax.py --network 's60' --patch '16_0_96_112_0' --loss-type 1 > logs60_l1_v4 2>&1 &
#python -u train_softmax.py --network 's60' --patch '0_0_96_95_0' --loss-type 1 --prefix '../model/spherefacex'
#python -u train_softmax.py --network 's20' --patch '0_0_96_112_0' --loss-type 0
#python -u train_softmax.py --network 'r50' --patch '0_0_96_112_0' --loss-type 0
#python -u train_softmax.py --network 'm4' --patch '0_0_96_112_0' --loss-type 1 --prefix '../model/spherefacem' --per-batch-size 224 > celm.log 2>&1 &
#python -u train_softmax.py --network 'm29' --patch '0_0_96_112_0' --loss-type 0 --lr 0.1 --prefix '../model/softmax' --verbose 2000 --per-batch-size 128 > sx_m29.log 2>&1 &
#python -u train_softmax.py --network 'm29' --patch '0_0_96_112_0' --loss-type 1 --lr 0.1 --prefix '../model/sphere47' --verbose 2000 --per-batch-size 224 --beta-min 4.7 > sp_m29_47.log 2>&1 &
export CUDA_VISIBLE_DEVICES='2,3'
#python -u train_softmax.py --network 'm1' --patch '0_0_96_112_0' --loss-type 0 --lr 0.01 --prefix '../model/marginal0' --verbose 2000
#python -u train_softmax.py --network 's60' --patch '0_0_96_95_0' --loss-type 1
#python -u train_softmax.py --network 's20' --patch '0_0_96_95_0' --loss-type 1
#python -u train_softmax.py --network 's60' --patch '0_0_96_112_0' --loss-type 1 --prefix '../model/spherefacec' > logs60_c 2>&1 &
#python -u train_marginal.py --patch '0_0_96_112_0' --network 's36' --verbose 1000 --lr 0.01
#python -u train_coco.py --patch '0_0_96_112_0' --images-per-identity 32
#python -u train_softmax.py --network 's36' --patch '0_0_96_112_0' --loss-type 1 --prefix '../model/spherefacei36' --per-batch-size 256
#python -u train_softmax.py --network 's36' --patch '0_0_96_112_0' --loss-type 1 --prefix '../model/spherefacei36' --per-batch-size 256 > cel4.log 2>&1 &
#python -u train_softmax.py --network 'm28' --patch '0_0_96_112_0' --loss-type 11 --lr 0.1 --prefix '../model/L11' --verbose 500 --per-batch-size 128 --images-per-identity 4
#python -u train_softmax.py --network 'm27' --patch '0_0_96_112_0' --loss-type 1 --lr 0.1 --prefix '../model/sphere' --verbose 2000 --per-batch-size 224 > sp_m27.log 2>&1 &
#python -u train_softmax.py --network 'm27' --patch '0_0_96_112_0' --loss-type 0 --lr 0.1 --prefix '../model/softmax' --verbose 2000 --per-batch-size 128 > sx_m27.log 2>&1 &
export CUDA_VISIBLE_DEVICES='4,5'
#python -u train_softmax.py --network 'm29' --patch '0_0_96_112_0' --loss-type 0 --lr 0.1 --prefix '../model/softmax' --verbose 2000 --per-batch-size 128 > sx_m29.log 2>&1 &
#python -u train_softmax.py --network 'm27' --patch '0_0_96_112_0' --loss-type 1 --lr 0.1 --prefix '../model/sphere' --verbose 2000 --per-batch-size 224 > sp_m27.log 2>&1 &
#python -u train_softmax.py --network 'm28' --patch '0_0_96_112_0' --loss-type 1 --lr 0.1 --prefix '../model/sphere' --verbose 2000 --per-batch-size 224 > sp_m28.log 2>&1 &
export CUDA_VISIBLE_DEVICES='6,7'
#python -u train_softmax.py --network 'm29' --patch '0_0_96_112_0' --loss-type 1 --lr 0.1 --prefix '../model/spherem' --verbose 2000 --per-batch-size 224
#python -u train_softmax.py --network 'm28' --patch '0_0_96_112_0' --loss-type 0 --lr 0.1 --prefix '../model/softmax' --verbose 2000 --per-batch-size 128 > sx_m28.log 2>&1 &
#python -u train_marginal.py --patch '0_0_96_112_0' --network 'i4' --verbose 2000 --lr 0.01
#python -u train_softmax.py --network 'i4' --patch '0_0_96_112_0' --loss-type 1 --gamma 0.06 --beta-min 4
#python -u train_softmax.py --network 'x4' --patch '0_0_96_112_0' --loss-type 1 --gamma 0.09
#python -u train_softmax.py --network 's60' --patch '0_0_80_95_0' --loss-type 1 > logs60_l1_v3 2>&1 &
#python -u train_softmax.py --network 's60' --patch '0_0_96_95_0' --loss-type 1 > logs60_l1_v2 2>&1 &
#python -u train_softmax.py --network 's20' --patch '0_0_96_112_0'
export CUDA_VISIBLE_DEVICES='4,5,6,7'
python -u train_softmax.py --network 'm29' --patch '0_0_96_112_0' --loss-type 0 --lr 0.1 --prefix '../model/softmax' --verbose 2000 --per-batch-size 128 > sx_m29.log 2>&1 &
#python -u train_softmax.py --network 's60' --patch '0_0_96_95_0' --loss-type 1 --gamma 0.06 --beta-freeze 5000 --prefix '../model/spherefacei' > cel2.log 2>&1 &
export CUDA_VISIBLE_DEVICES='0,1,2,3'
#python -u train_softmax.py --network 'm29' --patch '0_0_96_112_0' --loss-type 1 --lr 0.1 --prefix '../model/spherem' --verbose 2000 --per-batch-size 224 --lr-steps '60000,80000,90000' > spm_m29.log 2>&1 &
#python -u train_softmax.py --network 's60' --patch '0_15_96_112_0' --loss-type 1 --gamma 0.06 --beta-freeze 5000 --prefix '../model/spherefacei' > cel3.log 2>&1 &
export CUDA_VISIBLE_DEVICES='2'
#python -u train_marginal.py --patch '0_0_96_112_0' --network 's36' --verbose 2000 --lr 0.01 > mar_s36.log 2>&1 &
export CUDA_VISIBLE_DEVICES='3'
#python -u train_marginal.py --patch '0_0_96_112_0' --network 'i4' --verbose 2000 --lr 0.01 > mar_i4.log 2>&1 &
#python -u train_softmax.py --network 'i4' --patch '0_0_96_112_0' --loss-type 1 --gamma 0.06 --beta-freeze 5000
#python -u train_softmax.py --network 'r50' --patch '0_0_96_112_0' --loss-type 1 --gamma 0.24 > logr50_l1 2>&1 &
#python -u train_softmax.py --network 'r50' --patch '0_0_96_112_0' --loss-type 2 --verbose 100
#python -u train_softmax.py --network 'r50' --patch '0_0_96_95_0' > logr101_pu 2>&1 &
#python -u train_softmax.py --network 'r50' --patch '0_0_96_112_0'
#python -u train_softmax.py --network 'r101' --patch '0_0_96_95_0'
#python -u train_softmax.py --loss-type 1 --num-layers 64 --patch '0_0_96_112_0'
#python -u train_softmax.py --loss-type 1 --num-layers 36 --patch '0_0_96_95_0'
#python -u train_softmax.py --loss-type 1 --num-layers 20 --patch '0_0_80_95_0'

