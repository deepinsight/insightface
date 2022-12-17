
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=7 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=12345 train.py $@

