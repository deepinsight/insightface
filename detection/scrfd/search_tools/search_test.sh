#!/usr/bin/env bash


GPU=0
OUTPUT_DIR=wouts
THR=0.02
GROUP=scrfdgen2p5g
PREFIX=$GROUP

for i in {1..320}
do
    TASK="$PREFIX"_"$i"
    echo $TASK
    CUDA_VISIBLE_DEVICES="$GPU" python -u tools/test_widerface.py ./configs/"$GROUP"/"$TASK".py ./work_dirs/"$TASK"/latest.pth --mode 0 --thr "$THR" --out "$OUTPUT_DIR"/"$GROUP"/"$TASK"
done

