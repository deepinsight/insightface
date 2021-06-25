export CUDA_VISIBLE_DEVICES=0


nohup python3.7 valid.py \
    --network 'MobileFaceNet_128' \
    --checkpoint='emore_arcface' > "valid_log.log" 2>&1 &