export CUDA_VISIBLE_DEVICES=1

nohup python3.7 infer.py \
    --network 'MobileFaceNet_128' \
    --img='00000000.jpg' \
    --checkpoint 'emore_arcface' > "infer_log.log" 2>&1 &