
## I. CPP-Align
 
  -

## II. Face Mask Renderer

We provide a simple tool to add masks on face images automatically.

We can use this tool to do data augmentation while training our face recognition models.

| Face Image  | OP | Mask Image | Out |  
| ------- | ------ | --------- | ----------- |  
|  <img src="https://github.com/deepinsight/insightface/blob/master/python-package/insightface/data/images/Tom_Hanks_54745.png" alt="face" height="112" /> | +F  | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask1.jpg" alt="mask" height="112" />     | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask_out1.jpg?raw=true" alt="mask" height="112" />      | 
|  <img src="https://github.com/deepinsight/insightface/blob/master/python-package/insightface/data/images/Tom_Hanks_54745.png" alt="face" height="112" /> | +F  | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/black-mask.png" alt="mask" height="112" />     | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask_out3.jpg?raw=true" alt="mask" height="112" />      | 
|  <img src="https://github.com/deepinsight/insightface/blob/master/python-package/insightface/data/images/Tom_Hanks_54745.png" alt="face" height="112" /> | +H  | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask2.jpg?raw=true" alt="mask" height="112" />     | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask_out2h.jpg?raw=true" alt="mask" height="112" />      | 

**F** means FULL while **H** means HALF.

### Prepare

- insightface package library

   ``pip install -U insightface``

- insightface model pack

  ``bash> insightface-cli model.download antelope``
  
- BFM models

   Please follow the tutorial of [https://github.com/YadiraF/face3d/tree/master/examples/Data/BFM](https://github.com/YadiraF/face3d/tree/master/examples/Data/BFM) to generate `BFM.mat` and `BFM_UV.mat`. Put them into the insightface model pack directory, such as ``~/.insightface/models/antelope/``
   
   
- mask images

   some mask images are included in insightface package, such as 'mask\_blue', 'mask\_white', 'mask\_black' and 'mask\_green'.
   
### Add Mask to Face Image

Please refer to `make_renderer.py` for detail example. 

(1) init renderer:
```
import insightface
from insightface.app import MaskRenderer
tool = MaskRenderer()
tool.prepare(ctx_id=0, det_size=(128,128)) #use gpu
```

(2) load face and mask images
```
from insightface.data import get_image as ins_get_image
image = ins_get_image('Tom_Hanks_54745')
mask_image  = "mask_blue"
```

(3) build necessary params for face image, this can be done in offline.
```
params = tool.build_params(image)
```

(4) do mask render, it costs about `10ms` on 224x224 UV size, CPU single thread.
```
mask_out = tool.render_mask(image, mask_image, params)
```

(5) do half mask render.
```
mask_half_out = tool.render_mask(image, mask_image, params, positions=[0.1, 0.5, 0.9, 0.7])
```
