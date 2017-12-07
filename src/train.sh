#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/opt/jiaguo/faces_vgg_112x112

NETWORK=r49
JOB=softmax1e3
MODELDIR="../model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 0 --lr 0.1 --prefix "$PREFIX" --per-batch-size 128 --image-size '112,112' --version-input 1 --version-output E --version-unit 3 > "$LOGFILE" 2>&1 &

