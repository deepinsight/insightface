ip_list=("ip1" "ip2" "ip3" "ip4")

config=wf42m_pfc03_32gpu_r100

for((node_rank=0;node_rank<${#ip_list[*]};node_rank++));
do 
  ssh face@${ip_list[node_rank]} "cd `pwd`;PATH=$PATH \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=${#ip_list[*]} \
  --node_rank=$node_rank \
  --master_addr=${ip_list[0]} \
  --master_port=22345 train.py configs/$config" &
done
