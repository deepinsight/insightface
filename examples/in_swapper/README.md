# InsightFace Swapper

In this example, we provide one-line simple code for subject agnostic identity transfer from source face to the target face.

The input and output resolution of this tool is 128x128, which is obviously smaller than our [online demo](http://demo.insightface.ai:7009/). 

The network size and computation complexity are both very large, so do not use it in any product, but for academic purposes instead.


## Usage

Firstly install insightface python library, with version>=0.7:

```
pip install -U insightface
```

Then use `buffalo_l` recognition model and initialize the INSwapper class. 

Note that now we can only accept latent embedding from the `buffalo_l` arcface model, otherwise the result will be not normal.

The auto-downloading of `inswapper_128.onnx` may be distable if the network traffic is too high. 
You can also download from [googledrive](https://drive.google.com/file/d/1GW7Q41Uk4H30wVFL2Tl4Kl8MWIV4fVFC/view?usp=share_link) instead and put it under `~/.insightface/models/`.

For detailed code, please check the [example](inswapper_main.py).

## Result:

Input: 

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1.jpg" width="640" />

---Then we change the identity to Ross for all faces in this image.---

Direct Outputs:

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1_swapped2.jpg" width="640" />

Paste Back:

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1_swapped.jpg" width="640" />

