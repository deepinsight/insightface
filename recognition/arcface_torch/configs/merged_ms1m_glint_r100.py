import os

from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r100"
config.resume = False
config.output = "/output/merged_ms1m_glint_r100"
config.embedding_size = 512
config.sample_rate = 0.2
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 320
config.lr = 0.1
config.verbose = 2000
config.dali = False
config.num_workers = 24 # Try to change this according to the number of CPU cores, 12 is good for 16 cores, but if you have 32 cores, you can set it to 24 or 28

config.rec = "/datasets/merged_ms1m_glint"
config.num_classes = 453663
config.num_image = 22271167
config.num_epoch = 20
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.save_all_states = False# To save GPU memory, only save the state of backbone, not the state of partial fc and optimizer

config.using_wandb = True
config.wandb_key = os.getenv("WANDB_API_KEY")
config.wandb_entity = os.getenv("WANDB_ENTITY")
config.wandb_project = os.getenv("WANDB_PROJECT")
config.wandb_log_all = True
config.wandb_resume = False
config.suffix_run_name = "merged_ms1m_glint_r100"
config.notes = "Training r100 on merged MS1MV3 + Glint360K dataset"

