# Two-image recognition demo

[main.py](main.py) is a legacy local command-line demo. It detects one face in
each image and compares their ArcFace embeddings. For current library usage,
see the [Python package guide](../../python-package/README.md).

The script loads these two `buffalo_l` files directly; it does not download them:

```text
~/.insightface/models/buffalo_l/det_10g.onnx
~/.insightface/models/buffalo_l/w600k_r50.onnx
```

Its local modules use NumPy, OpenCV, ONNX, ONNX Runtime, and scikit-image. The
loaders explicitly request `CUDAExecutionProvider`, so the runtime and CUDA
dependencies must match the environment used to run it.

From the repository root, pass the two image paths as positional arguments:

```bash
python web-demos/src_recognition/main.py /path/to/image1.jpg /path/to/image2.jpg
```

The script prints `sim: <cosine similarity>, message: <comparison message>`.
Its fixed demo thresholds are 0.2 and 0.28; when no face is detected, it prints
`sim: -1.0000` with `Face not found in Image-1` or `Face not found in Image-2`.
