export CUDA_VISIBLE_DEVICES=1

log_name="log"


# bs is 512 for best
nohup python3.7 train.py \
    --network 'MobileFaceNet_128' \
    --lr=0.1 \
    --batch_size 256 \
    --weight_decay 2e-4 \
    --embedding_size 128 \
    --logdir="${log_name}" \
    --output "emore_arcface" > "log.log" 2>&1 &
