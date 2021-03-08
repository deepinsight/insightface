## Eval IJBC

```shell
CUDA_VISIBLE_DEVICES=0,1 python eval_ijbc.py \
--model-prefix tmp_models/backbone.pth \
--image-path /data/anxiang/IJB_release/IJBC \
--result-dir result \
--batch-size 128 \
--job cosface \
--target IJBC \
--network iresnet50
```

## Eval MegaFace

