## Eval IJBC

```shell
# model-prefix       your model path
# image-path         your IJBC path
# result-dir         your result path
# network            your backbone
CUDA_VISIBLE_DEVICES=0,1 python eval_ijbc.py \
--model-prefix ms1mv3_arcface_r50/backbone.pth \
--image-path IJB_release/IJBC \
--result-dir ms1mv3_arcface_r50 \
--batch-size 128 \
--job ms1mv3_arcface_r50 \
--target IJBC \
--network iresnet50
```

## Eval MegaFace
pass

