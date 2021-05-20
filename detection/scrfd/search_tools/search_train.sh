#!/usr/bin/env bash

GROUP=scrfdgen2.5g
TASKS_PER_GPU=8
OFFSET=1
for i in {0..7}
do
    let a=TASKS_PER_GPU*i+OFFSET
    let i2=i+1
    let b=TASKS_PER_GPU*i2+OFFSET
    echo $i,$a,$b,$GROUP
    python -u search_tools/search_train.py $i $a $b $GROUP > "gpu$i".log 2>&1 &
done

