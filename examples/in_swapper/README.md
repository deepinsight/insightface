# InsightFace Swapper

## Update

Please use our discord web demo instead. This python example is temporarily removed.


In this example, we provide one-line simple code for subject agnostic identity transfer from source face to the target face.

The input and output resolution of this tool is 128x128.


## Usage

Firstly install insightface python library, with version>=0.7:

```
pip install -U insightface
```

Second, download the `inswapper_128.onnx` swapping model from [googledrive]() and put it under `~/.insightface/models/`.

Then use the recognition model from our `buffalo_l` pack and initialize the INSwapper class. 

Note that now we can only accept latent embedding from the `buffalo_l` arcface model, otherwise the result will be not normal.

For detail code, please check the [example](inswapper_main.py).

## Result:

Input: 

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1.jpg" width="640" />

---Then we change the identity to Ross for all faces in this image.---

Direct Outputs:

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1_swapped2.jpg" width="640" />

Paste Back:

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1_swapped.jpg" width="640" />

