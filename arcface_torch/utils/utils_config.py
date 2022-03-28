import importlib
import os.path as osp


def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    # temp_config_name = osp.basename(config_file)
    # temp_module_name = osp.splitext(temp_config_name)[0]

    temp_module_name = osp.splitext(config_file)[0]
    temp_module_name = temp_module_name.replace("/", ".")
    config = importlib.import_module("configs.base")
    cfg = config.config
    # config = importlib.import_module("configs.scale.%s" % temp_module_name)
    config = importlib.import_module(temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfg