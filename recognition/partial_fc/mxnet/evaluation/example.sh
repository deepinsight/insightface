python -u ijb.py \
--model-prefix /anxiang/opensource/model/celeb360k_final0.1/model \
--image-path /data/anxiang/datasets/IJB_release/IJBC \
--result-dir /anxiang/opensource/results/test \
--model-epoch 0 \
--gpu 0,1,2,3,4,5,6,7 \
--target IJBC \
 --job cosface \
--batch-size 256