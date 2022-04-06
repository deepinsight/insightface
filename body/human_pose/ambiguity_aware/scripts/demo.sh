#!/bin/bash

rm -rf demo_output

python inference.py --indir demo_input --outdir demo_output --cfg ../cfg/h36m_gt_scale.yaml --pretrain ../models/tmc_klbone.pth.tar
