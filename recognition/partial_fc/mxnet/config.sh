export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_ALLGATHER=NCCL
export HOROVOD_GPU_BROADCAST=NCLL
export MXNET_CPU_WORKER_NTHREADS=3

# use `which python` to get the absolute path of your python interpreter
#
PYTHON_EXEC=/usr/bin/python
${PYTHON_EXEC} train_memory.py \
--dataset glint360k_8GPU \
--loss cosface \
--network r100 \
--models-root /data/anxiang/opensource/glint360k_8GPU_r100FC_1.0_fp32_cosface
