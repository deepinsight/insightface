#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=~/datasets/glintasia
PRETRAINED=~/deeplearning/insightface/model-r100-ii-1-15/model,5

NETWORK=r100
JOB=ii-1-16
MODELDIR="../model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --lr-steps=21,40000,80000  --verbose 2000 --pretrained $PRETRAINED --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 96 > "$LOGFILE" 2>&1
#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --lr-steps=50000,80000,11000 --verbose 2000 --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 96 > "$LOGFILE" 2>&1

#CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --lr-steps=10000,50000,80000,11000 --target=cfp_ff --verbose 2000 --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 64 > "$LOGFILE" 2>&1

CUDA_VISIBLE_DEVICES='' python -u train_softmax.py --lr-steps=50000,100000,14000 --target=cfp_ff --verbose 2000 --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 512