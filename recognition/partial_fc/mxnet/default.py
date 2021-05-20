from easydict import EasyDict as edict

config = edict()
# loss
config.embedding_size = 512
config.bn_mom = 0.9
config.workspace = 256
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_output = 'FC'
config.frequent = 20
config.verbose = 2000
config.image_size = 112
config.memonger = False

config.debug = 0
config.fp16 = False
config.batch_size = 64
config.backbone_lr = 0.1
config.memory_bank_lr = config.backbone_lr
config.sample_ratio = 1.0


def generate_config(loss_name, dataset, network):

    # loss
    if loss_name == 'arcface':
        config.loss_s = 64.0
        config.loss_m1 = 1.0
        config.loss_m2 = 0.5
        config.loss_m3 = 0.0
    elif loss_name == 'cosface':
        config.loss_s = 64.0
        config.loss_m1 = 1.0
        config.loss_m2 = 0.0
        config.loss_m3 = 0.4

    # dataset
    if dataset == 'webface':
        config.lr_steps = '20000,28000'
        config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
        config.rec = '/anxiang/datasets/webface/train.rec'
        config.rec = '/train_tmp/webface/train.rec'
        config.num_classes = 10575
        config.max_update = 32000

    # glint360k 17091657
    # md5sum:
    # 5d9cd9f262ec87a5ca2eac5e703f7cdf train.idx
    # 8483be5af6f9906e19f85dee49132f8e train.rec

    # make training faster
    # our RAM is 256G
    # mount -t tmpfs -o size=140G  tmpfs /train_tmp

    elif dataset == 'glint360k_8GPU':
        config.lr_steps = '200000,400000,500000,550000'
        config.val_targets = [
            'agedb_30', 'calfw', 'cfp_ff', 'cfp_fp', 'cplfw', 'lfw', 'vgg2_fp'
        ]
        config.rec = '/train_tmp/glint360k/train.rec'
        config.num_classes = 360232
        config.batch_size = 64
        config.max_update = 600000

    elif dataset == 'glint360k_16GPU':
        config.lr_steps = '200000,280000,360000'
        config.val_targets = ['agedb_30', 'cfp_fp', 'lfw']
        config.rec = '/train_tmp/glint360k/train.rec'
        config.num_classes = 360232
        config.max_update = 400000

    elif dataset == 'emore':
        config.lr_steps = '100000,160000'
        config.val_targets = ['agedb_30', 'cfp_fp', 'lfw']
        config.rec = '/anxiang/datasets/faces_emore/train.rec'
        config.rec = '/train_tmp/faces_emore/train.rec'
        config.num_classes = 85742
        config.batch_size = 64
        config.max_update = 180000

    elif dataset == '100w':
        config.debug = 1
        config.num_classes = 100 * 10000
        config.lr_steps = '20000,28000'
        config.max_update = 32000

    elif dataset == '1000w':
        config.debug = 1
        config.num_classes = 1000 * 10000
        config.lr_steps = '20000,28000'
        config.max_update = 32000

    elif dataset == '2000w':
        config.debug = 1
        config.num_classes = 2000 * 10000
        config.lr_steps = '20000,28000'
        config.max_update = 32000

    elif dataset == '3000w':
        config.debug = 1
        config.num_classes = 3000 * 10000
        config.lr_steps = '20000,28000'
        config.max_update = 32000

    elif dataset == '10000w':
        config.debug = 1
        config.num_classes = 10000 * 10000
        config.lr_steps = '20000,28000'
        config.max_update = 32000

    # network
    if network == 'r100':
        config.net_name = 'resnet'
        config.num_layers = 100
    elif network == 'r122':
        config.net_name = 'resnet'
        config.num_layers = 122
    elif network == 'r50':
        config.net_name = 'resnet'
        config.num_layers = 50
    elif network == 'rx101':
        config.net_name = 'fresnext'
        config.num_layers = 101
    elif network == 'rx50':
        config.net_name = 'fresnext'
        config.num_layers = 50
