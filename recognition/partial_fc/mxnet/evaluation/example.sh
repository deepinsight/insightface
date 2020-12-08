#!/bin/bash

# run `python ijb.py --help` for more information
python -u ijb.py \
--model-prefix ./models/y1-cosface-glink360/model \
--image-path /data/IJB_release/IJBC \
--result-dir ./results/test \
--model-epoch 0 \
--gpu 0,1,2,3,4,5,6,7 \
--target IJBC \
--job cosface \
--batch-size 256 \
-es 128

