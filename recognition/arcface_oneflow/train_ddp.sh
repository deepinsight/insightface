# set -aux

MASTER_ADDR=127.0.0.1
MASTER_PORT=17788
DEVICE_NUM_PER_NODE=8
NUM_NODES=1
NODE_RANK=0


export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
export NCCL_DEBUG=INFO
export ONEFLOW_DEBUG_MODE=True


NCCL_DEBUG=INFO \
python3 -m oneflow.distributed.launch \
--nproc_per_node $DEVICE_NUM_PER_NODE \
--nnodes $NUM_NODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
train.py configs/ms1mv3_r50.py 
