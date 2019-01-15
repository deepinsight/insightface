#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=~/datasets/glintasia
PRETRAINED=~/insightface/model-r100-ii/model,0

NETWORK=r100
JOB=ii
MODELDIR="../model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
#CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --verbose 40 --pretrained $PRETRAINED --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 32 > "$LOGFILE" 2>&1
CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --verbose 10000 --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 96 > "$LOGFILE" 2>&1

