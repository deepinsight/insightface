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


## Result

| Datasets       | Backbone            | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) |
|:---------------|:--------------------|:------------|:------------|:------------|
| WF12M-PFC-0.05 | r100                | 94.05       | 97.51       | 95.75       |
| WF12M-PFC-0.1  | r100                | 94.49       | 97.56       | 95.92       |
| WF12M-PFC-0.2  | r100                | 94.75       | 97.60       | 95.90       |
| WF12M-PFC-0.3  | r100                | 94.71       | 97.64       | 96.01       |
| WF12M          | r100                | 94.69       | 97.59       | 95.97       |