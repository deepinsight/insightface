## Eval on ICCV2021-MFR

coming soon.


## Eval IJBC
You can eval ijbc with pytorch or onnx.


1. Eval IJBC With Onnx
```shell
CUDA_VISIBLE_DEVICES=0 python onnx_ijbc.py --model-root ms1mv3_arcface_r50 --image-path IJB_release/IJBC --result-dir ms1mv3_arcface_r50
```

2. Eval IJBC With Pytorch
```shell
CUDA_VISIBLE_DEVICES=0,1 python eval_ijbc.py \
--model-prefix ms1mv3_arcface_r50/backbone.pth \
--image-path IJB_release/IJBC \
--result-dir ms1mv3_arcface_r50 \
--batch-size 128 \
--job ms1mv3_arcface_r50 \
--target IJBC \
--network iresnet50
```

## Inference

```shell
python inference.py --weight ms1mv3_arcface_r50/backbone.pth --network r50
```
