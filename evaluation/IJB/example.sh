#!/usr/bin/env bash

python -u IJB_11.py --model-prefix ./pretrained_models/r100-arcface/model --model-epoch 1 --gpu 0 --target IJBC --job arcface > ijbc_11.log 2>&1 &

python -u IJB_1N.py --model-prefix ./pretrained_models/r100-arcface/model --model-epoch 1 --gpu 0 --target IJBB --job arcface > ijbb_1n.log 2>&1 &

