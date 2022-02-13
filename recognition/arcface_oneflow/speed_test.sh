# set -aux


MASTER_ADDR=127.0.0.1
MASTER_PORT=17788
DEVICE_NUM_PER_NODE=1 
NUM_NODES=1
NODE_RANK=0


export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
export NCCL_DEBUG=INFO
export ONEFLOW_DEBUG_MODE=False


NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m oneflow.distributed.launch \
--nproc_per_node $DEVICE_NUM_PER_NODE \
--nnodes $NUM_NODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
train.py configs/speed.py  --train_num 400 --batch_size 128 --graph --channel_last True --fp16 True
