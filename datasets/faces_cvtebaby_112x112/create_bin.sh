#!/bin/bash

python gen_valid_landmark.py
python align_cvtebaby.py
python cvtebaby2pack.py --output /data/victor/insightface/datasets/faces_cvtebaby_112x112/cvte_baby9000.bin
