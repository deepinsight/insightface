#!/bin/bash

mpirun -np 8 \
-hostfile hosts/host_8 \
-x HOROVOD_CACHE_CAPACITY=8096 \
-bind-to none -map-by slot \
-x LD_LIBRARY_PATH -x PATH \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include eth1 \
-x OMP_NUM_THREADS=2 \
-x MXNET_USE_OPERATOR_TUNING=1 \
-x MXNET_USE_NUM_CORES_OPERATOR_TUNING=1 \
-x MXNET_CUDNN_AUTOTUNE_DEFAULT=1 \
bash config.sh
