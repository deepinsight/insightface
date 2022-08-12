import importlib
import os
import os.path as osp
import numpy as np

def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    #print('A:', config_file, temp_config_name, temp_module_name)
    config1 = importlib.import_module("configs.base")
    importlib.reload(config1)
    cfg = config1.config
    #print('B1:', cfg)
    config2 = importlib.import_module("configs.%s"%temp_module_name)
    importlib.reload(config2)
    #reload(config2)
    job_cfg = config2.config
    #print('B2:', job_cfg)
    cfg.update(job_cfg)
    cfg.job_name = temp_module_name
    #print('B:', cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    #print('C:', cfg.output)
    cfg.flipindex = np.load(cfg.flipindex_file)
    return cfg
