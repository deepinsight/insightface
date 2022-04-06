
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=12345 train.py $@

ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
