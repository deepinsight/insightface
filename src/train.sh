#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/home/lijc08/datasets/glintasia/faces_glintasia

NETWORK=r50
JOB=softmax1e3
MODELDIR="../model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 0 --prefix "$PREFIX" --per-batch-size 64
#> "$LOGFILE" 2>&1 &

