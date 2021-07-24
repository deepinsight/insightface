export CUDA_VISIBLE_DEVICES=1

log_name="log"


# If you want to reduce batchsize because of GPU memory,
# you can reduce batch size and lr proportionally.
python3.7 train.py \
    --network 'MobileFaceNet_128' \
    --lr=0.1 \
    --batch_size 16 \
    --weight_decay 2e-4 \
    --embedding_size 128 \
    --logdir="${log_name}" \
    --output "emore_arcface" \
    --is_bin=False
