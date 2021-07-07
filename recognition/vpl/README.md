
# Variational Prototype Learning for Deep Face Recognition

This is the Pytorch implementation of our paper  [Variational Prototype Learning for Deep Face Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Variational_Prototype_Learning_for_Deep_Face_Recognition_CVPR_2021_paper.pdf) which is accepted by CVPR-2021.

## How to run

Define a new configure file such as `configs/example_ms1m.py`, and start the training process by:

``
bash run.sh configs/example_ms1m.py
``

## Results

Results on WebFace600K(subset of WebFace260M), loss is margin-based softmax.

| Backbone   | Dataset    | VPL? | Mask   | Children | African | Caucasian | South Asian | East Asian | MR-All    | 
|------------|------------|------------|--------|----------|---------|-----------|-------------|------------|--------|
| R50  | WebFace600K  | NO | 78.949 | 74.772   | 89.231  | 94.114    | 92.308      | 73.765     | 90.591 | 
| R50  | WebFace600K  | YES | 78.884 | 75.739   | 89.424  | 94.220    | 92.609      | 74.365     | 90.942 | 
