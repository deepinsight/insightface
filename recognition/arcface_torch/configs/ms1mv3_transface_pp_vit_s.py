from easydict import EasyDict as edict

config = edict()
config.network = "transface_pp_vit_s"
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

# TransFace++ (TPAMI 2025): the model consumes image bytes, not pixels
config.byte_format = "tiff"     # tiff (paper best) | png | fhwc | fchw; file bytes are built on the GPU
config.use_topology = True      # persistent-homology feature cross-attended in the last block
config.tibc_prob = 0.3          # topology-based image bytes compression, applied per sample with this prob.
config.ehsm = True              # entropy-guided hard sample mining (loss re-weighting)
config.ehsm_gamma = 1.0
config.loss = "arcface"         # arcface | adaface

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 20
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
