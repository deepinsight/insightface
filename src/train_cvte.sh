#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=../datasets/faces_cvtebaby_112x112
#DATA_DIR=../datasets/faces_ms1m_112x112

# NETWORK=r100
# #NETWORK=r50
# JOB=triplet_cvtebaby
# LR_STEPS='200,400,600'
# VERBOSE=1 #100
# TEST_TARGET='cvte_baby9000'
# TRAINED_MODEL=../models/r100-combined-margin/model-r100-combined-margin,285
# #TRAINED_MODEL=../models/r100-small-lr/model-r100-small-lr,662
# #TRAINED_MODEL=../models/model-r50-am-lfw/model,0
# MODELDIR="../models/model-$NETWORK-$JOB"
# mkdir -p "$MODELDIR"
# PREFIX="$MODELDIR/model"
# LOGFILE="$MODELDIR/log"
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --data-dir $DATA_DIR --network $NETWORK \
# --loss-type 12 --lr 0.005 --mom 0.0 --prefix "$PREFIX" --per-batch-size 60 --pretrained "$TRAINED_MODEL" \
# --lr-step "$LR_STEPS" --target "$TEST_TARGET" --verbose "$VERBOSE" --triplet-bag-size 3600 > "$LOGFILE" 2>&1 &


NETWORK=r100  # 过拟合训练分类精度100%，验证集验证精度85%
JOB=cvtebaby
LR_STEPS='200,400,600'
VERBOSE=100
TEST_TARGET='cvte_baby9000'
TRAINED_MODEL=../models/r100-combined-margin/model-r100-combined-margin,285
MODELDIR="../models/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network $NETWORK --loss-type 4 \
--margin-m 0.5 --data-dir ../datasets/faces_cvtebaby_112x112/  --prefix ../model-r100 \
--per-batch-size 64 --lr-step "$LR_STEPS" --pretrained "$TRAINED_MODEL" --finetune True \
--target "$TEST_TARGET" --verbose "$VERBOSE" --lr 0.0001 > "$LOGFILE" 2>&1 &


tail -f "$LOGFILE"

