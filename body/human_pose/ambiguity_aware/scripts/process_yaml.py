#!/usr/bin/env python
# coding=utf-8

import yaml 
import os 
import os.path as osp 

for _root, _dirs, _files in os.walk("./"): 
    for _file in _files: 
        if not _file.endswith(".yaml"): 
            continue
        filepath = osp.join(_root, _file)
        with open(filepath, "r") as f: 
            data = yaml.load(f)

        loss_weights = data['TRAIN']['LOSS_WEIGHTS']
        if len(loss_weights) == 5: 
            data['TRAIN']['LOSS_WEIGHTS'] = loss_weights[:4]

        with open(filepath, "w") as f: 
            yaml.dump(data, f)
