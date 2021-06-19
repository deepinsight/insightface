#!/usr/bin/env bash

python -u ijb_11.py --model-prefix ./pretrained_models/r100-arcface/model --model-epoch 1 --gpu 0 --target IJBC --job arcface > ijbc_11.log 2>&1 &

python -u ijb_1n.py --model-prefix ./pretrained_models/r100-arcface/model --model-epoch 1 --gpu 0 --target IJBB --job arcface > ijbb_1n.log 2>&1 &

