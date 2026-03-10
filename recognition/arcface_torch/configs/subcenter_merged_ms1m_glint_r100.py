import os

from easydict import EasyDict as edict

config = edict()

# Sub-center ArcFace settings
config.margin_list = (1.0, 0.5, 0.0)  # (m1, m2, m3) — standard ArcFace margins
config.num_subcenters = 3              # K sub-centers per class (paper default: 3)

config.network = "r100"
config.resume = False
config.output = "/output/subcenter_merged_ms1m_glint_r100"
config.embedding_size = 512
config.sample_rate = 0.2
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 320
config.lr = 0.1
config.verbose = 10000
config.dali = False
config.num_workers = 3 # Keep low to fit alongside other processes

config.rec = "/datasets/merged_ms1m_glint_rec"
config.num_classes = 453663
config.num_image = 22271167
config.num_epoch = 30
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.save_all_states = True # Save full checkpoint with backbone + optimizer states (not just best_model.pt)
config.save_every_epoch = 5   # Save a model snapshot every N epochs (0 to disable)

config.using_wandb = True
config.wandb_key = os.getenv("WANDB_API_KEY")
config.wandb_entity = os.getenv("WANDB_ENTITY")
config.wandb_project = os.getenv("WANDB_PROJECT")
config.wandb_log_all = True
config.wandb_resume = False
config.suffix_run_name = "subcenter_merged_ms1m_glint_r100_2nd_try"
config.notes = "Sub-center ArcFace (K=3) r100 on merged MS1MV3 + Glint360K dataset"
