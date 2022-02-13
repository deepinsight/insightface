#!/bin/bash

mpirun -np 8 \
-hostfile hosts/host_8 --allow-run-as-root \
-bind-to none -map-by slot \
-x LD_LIBRARY_PATH -x PATH \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include eth0 \
-x OMP_NUM_THREADS=2 \
bash config.sh
