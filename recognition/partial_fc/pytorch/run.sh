# /usr/bin/
export OMP_NUM_THREADS=4
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 partial_fc.py | tee hist.log
ps -ef | grep "partial_fc" | grep -v grep | awk '{print "kill -9 "$2}' | sh
