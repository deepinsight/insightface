from easydict import EasyDict as edict

config = edict()
config.network = "transface_vit_l"
config.resume = False
config.output = None

config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
config.lr = 1e-3
config.verbose = 2000
config.dali = False
config.optimizer = "adamw"

# TransFace recipe (paper / official code): DPAP + EHSM
config.dpap_prob = 0.2      # fraction of images whose dominant patches are perturbed
config.dpap_topk = 7        # K0 dominant patches (by SE weight) per image
config.dpap_alpha = 1.0     # lam ~ U(0, alpha) amplitude mixing strength
config.ehsm = True          # entropy-guided hard sample mining
config.ehsm_gamma = 1.0
config.loss = "arcface"     # arcface | adaface

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 35
config.warmup_epoch = 3
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
