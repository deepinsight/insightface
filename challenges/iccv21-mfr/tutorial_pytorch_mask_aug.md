# A tutorial on how to enable mask augmentation on arcface_torch training.

The python package insightface==0.3.2 provides utilities to enable mask augmentation within one line:

```
transform_list.append(
   MaskAugmentation(
      mask_names=['mask_white', 'mask_blue', 'mask_black', 'mask_green'], 
      mask_probs=[0.4, 0.4, 0.1, 0.1], h_low=0.33, h_high=0.4, p=self.mask_prob)
   )
```

### Prepare

1. Download antelope model pack by `bash> insightface-cli model.download antelope` which will be located at `~/.insightface/models/antelope`
2. Generate BFM.mat and BFM_UV.mat following [here](https://github.com/deepinsight/insightface/tree/master/recognition/tools#data-prepare), for license concern.
3. Generate new mask-rec dataset by `bash> insightface-cli rec.addmaskparam /data/ms1m-retinaface-t1 /data/ms1m-retinaface-t1mask` which generates and writes the mask params of each image into the record.


### Add Mask Renderer Augmentation
just by following code:
```
from insightface.app import MaskAugmentation
self.transform_list.append(
    MaskAugmentation(
    mask_names=['mask_white', 'mask_blue', 'mask_black', 'mask_green'], 
    mask_probs=[0.4, 0.4, 0.1, 0.1], 
    h_low=0.33, h_high=0.4, p=0.1)
)
```

Please check [dataset_mask.py](https://github.com/deepinsight/insightface/blob/master/challenges/iccv21-mfr/dataset_mask.py) for detail. 

You can override the original dataset.py with this file to simply enable mask augmentation.
