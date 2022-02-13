# Person Detection

This person detection example is built by [SCRFD](../../detection/scrfd) approch.

## Usage

Firstly install insightface python library:

```
pip install -U insightface
```

and then load our person detection model by:

```
detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
```

the model will be auto-downloaded from our storage server.

## Detection Result:

In this example, we support full-body detection and recognize the corresponding visible region in a single forward pass.

Please see [scrfd_person.py](scrfd_person.py) for detail on how to visualize the results.

The green bounding box shows the full-body while the blue mask indicates the visible region.

(We make tests on the input size of 640)

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/283554,c2d0000d40862ba.jpg" width="640" />

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/283647,18e170005675c161.jpg" width="640" />

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/283554,2290700005b7d575.jpg" width="640" />

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/283554,175820000e7255da.jpg" width="640" />
