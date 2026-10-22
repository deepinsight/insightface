# Gender and Age Inference Example

[test.py](test.py) uses `FaceAnalysis` with the detection and gender/age modules on the bundled `t1` sample image. Install the [InsightFace Python library](../../python-package/README.md#installation) first, then run from the repository root:

```bash
python attribute/gender_age/test.py --ctx -1
```

`--ctx` defaults to `0`; a negative value forces CPU inference. Otherwise, acceleration depends on the available ONNX Runtime providers. The default model package is downloaded on first use if it is not already cached.

The script checks that six faces are detected, then prints each face's bounding box, predicted sex, and estimated age. It does not save an annotated image.

See [face attributes](../README.md) for the module overview and [datasets](../_datasets_/README.md) for dataset references.
