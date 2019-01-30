#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=2
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/opt/glintasia
PRETRAINED=~/deeplearning/insightface/model-r34-1-29/model,89

NETWORK=r34
JOB=1-30
MODELDIR="../model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --lr-steps=21,40000,80000  --verbose 2000 --pretrained $PRETRAINED --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 96 > "$LOGFILE" 2>&1
#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --lr-steps=50000,80000,11000 --verbose 2000 --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 96 > "$LOGFILE" 2>&1

#CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --lr-steps=10000,50000,80000,11000 --target=cfp_ff --verbose 2000 --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 64 > "$LOGFILE" 2>&1

#CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --lr=0.05 --lr-steps=1000,160000,280000 --target=cfp_ff --verbose 2000 --pretrained $PRETRAINED --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 80 > "$LOGFILE" 2>&1
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --lr=0.0005 --lr-steps=160000 --target=cfp_ff --verbose 2000 --pretrained $PRETRAINED --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 80 > "$LOGFILE" 2>&1
