
## I. CPP-Align
 
  -

## II. Face Mask Renderer

We provide a simple tool to add masks on face images automatically.

We can use this tool to do data augmentation while training our face recognition models.

| Face Image  | OP | Mask Image | Out |  
| ------- | ------ | --------- | ----------- |  
|  <img src="https://github.com/deepinsight/insightface/blob/master/deploy/Tom_Hanks_54745.png" alt="face" height="112" /> | +F  | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask1.jpg" alt="mask" height="112" />     | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask_out1.jpg?raw=true" alt="mask" height="112" />      | 
|  <img src="https://github.com/deepinsight/insightface/blob/master/deploy/Tom_Hanks_54745.png" alt="face" height="112" /> | +F  | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/black-mask.png" alt="mask" height="112" />     | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask_out3.jpg?raw=true" alt="mask" height="112" />      | 
|  <img src="https://github.com/deepinsight/insightface/blob/master/deploy/Tom_Hanks_54745.png" alt="face" height="112" /> | +H  | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask2.jpg?raw=true" alt="mask" height="112" />     | <img src="https://github.com/nttstar/insightface-resources/blob/master/images/mask_out2h.jpg?raw=true" alt="mask" height="112" />      | 

**F** means FULL while **H** means HALF.

### Data Prepare

- face3d library

   Please copy and install [`face3d`](https://github.com/YadiraF/face3d) library and put it under `face3d/` directory. Then make sure you can `import face3d` in python env.
  
- BFM models

   Please follow the tutorial of [https://github.com/YadiraF/face3d/tree/master/examples/Data/BFM](https://github.com/YadiraF/face3d/tree/master/examples/Data/BFM) to generate `BFM.mat` and `BFM_UV.mat`.
   
- 3D68 model

   Download our 3D68 pretrained model(if1k3d68-0000.params) from [baiducloud](https://pan.baidu.com/s/1f9UtTpaW4l65DMRZ5wmJew)(passwd:2zmz) or [googledrive](https://drive.google.com/file/d/1Tv9f6Iqm4423qEDdDwbyjk00plEfeG8a/view?usp=sharing) and put it under `assets_mask/`. Please also put the above BFM data model into this directory.
   
- mask images

   You can get example mask images by checking the image url from the table above.
   
### Add Mask to Face Image

Please refer to `make_renderer.py` for detail example. 

(1) init renderer:
```
tool = MaskRenderer('./assets_mask')
```

(2) load face and mask images
```
image = cv2.imread("../../deploy/Tom_Hanks_54745.png")
mask_image  = cv2.imread("masks/mask2.jpg")
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
