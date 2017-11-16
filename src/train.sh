#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/opt/jiaguo/faces_normed

CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_softmax.py --data-dir $DATA_DIR --network 'm29' --patch '0_0_96_112_0' --loss-type 0 --lr 0.1 --prefix '../model/softmax' --verbose 2000 --per-batch-size 128 > sx_m29.log 2>&1 &
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network 'm29' --patch '0_0_96_112_0' --loss-type 1 --lr 0.1 --prefix '../model/spherem' --verbose 2000 --per-batch-size 224 --lr-steps '60000,80000,90000' > spm_m29.log 2>&1 &

